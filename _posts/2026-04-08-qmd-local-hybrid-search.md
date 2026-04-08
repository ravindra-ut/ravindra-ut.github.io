---
layout: post
title: "QMD: Local Hybrid Search for the Age of Agents"
date: 2026-04-08
categories: [AI, Systems, Search]
tags: [search, retrieval, bm25, embeddings, reranking, mcp, agents]
---

*How a local search engine combines keyword matching, vector similarity, and LLM reranking to give agents access to your files — with no cloud dependency.*

## The Retrieval Problem Nobody Solved Cleanly

You have a folder of markdown notes: meeting transcripts, docs, a knowledge base. You want an LLM agent to use them. The obvious approaches each fail in a specific way.

**Keyword search (BM25)** finds documents that contain your exact words. Search "CUDA out of memory" and it finds every file mentioning that phrase. But search "GPU crash during training" and it finds nothing, because no document uses those words even though several describe that exact problem. BM25 is precise when you know the vocabulary. Useless when you don't.

**Semantic search (vector embeddings)** understands meaning. "GPU crash during training" retrieves documents about CUDA OOM errors because the embeddings are close. But ask for "error code 0x80070057" and it returns vaguely related documents about errors in general, because embedding models compress specific identifiers into fuzzy semantic neighborhoods. Semantic search captures intent. It loses precision.

**LLM reranking** reads each candidate result and judges relevance using full language understanding. It's the most accurate method. It's also the slowest, by orders of magnitude. You can't rerank your entire corpus for every query.

Each method fails where another succeeds. The right answer is to use all three, in sequence, each compensating for the last. [QMD](https://github.com/tobi/qmd), built by Tobi Lütke, does exactly this — and runs it all locally.

## The Architecture: Three Stages, One Query

QMD implements a retrieval pipeline with three backends that fuse their results:

```
Query
  │
  ├──→ BM25 (keyword match)      ──→ top results + scores
  │
  ├──→ Vector search (embeddings) ──→ top results + scores
  │
  └──→ [optional] LLM expansion  ──→ rewritten query variants
                                        │
                                        └──→ fed back into BM25 + vector
         │
         ▼
   Score normalization ([0, 1])
         │
         ▼
   Reciprocal Rank Fusion
         │
         ▼
   LLM Reranking (local GGUF model)
         │
         ▼
   Final ranked results
```

Each stage is doing something the others can't.

### Stage 1: Dual Retrieval

BM25 and vector search run in parallel against the same corpus. BM25 uses traditional inverted-index full-text matching — fast, exact, and oblivious to meaning. Vector search embeds the query and finds chunks with nearby embeddings — slower, approximate, and oblivious to exact phrasing.

The two result sets overlap partially. BM25 finds documents the vector search missed (exact identifiers, code snippets, error messages). Vector search finds documents BM25 missed (paraphrased concepts, related topics). The union is more useful than either alone.

QMD uses EmbeddingGemma by default for embeddings, running locally through node-llama-cpp with GGUF quantized models. No API calls. No data leaving your machine.

### Stage 2: Score Fusion

Two ranked lists from two different scoring functions. BM25 scores are unbounded term-frequency numbers. Vector scores are cosine similarities between 0 and 1. You can't compare them directly.

QMD normalizes both to a [0, 1] range, then combines them using **Reciprocal Rank Fusion (RRF)**. RRF is a simple formula that works surprisingly well:

```
RRF_score(doc) = Σ  1 / (k + rank_i(doc))
                 i
```

For each ranking system `i`, take the document's rank, add a constant `k` (typically 60), and invert. Sum across all systems. A document ranked #1 in both lists gets a higher fused score than a document ranked #1 in one and absent from the other.

RRF has a useful property: it doesn't care about the raw scores, only the ordering. This makes it robust to the wildly different score distributions that BM25 and vector search produce. A document that both systems think is relevant floats to the top. A document that only one system likes still appears, but lower.

### Stage 3: LLM Reranking

The fused list is good but noisy. Both retrieval stages use shallow matching — statistical co-occurrence for BM25, geometric proximity for vectors. Neither actually *reads* the documents.

QMD passes the top candidates through a local reranker (Qwen3-Reranker by default) that scores each document against the query using full cross-attention. The reranker sees the query and the document together, not as independent embeddings, so it can catch relevance that depends on the relationship between them.

This is computationally expensive. That's why it runs last, on a small candidate set. The earlier stages do the heavy lifting of narrowing thousands of chunks to dozens. The reranker only needs to sort those dozens correctly.

## What Makes the Chunking Non-Trivial

Before any of this works, documents need to be split into chunks. This is the part most retrieval systems get lazy about: split on paragraph boundaries or every N tokens and call it done.

QMD does two things differently. For prose, it uses regex-based splitting that respects document structure: headings, section breaks, paragraph boundaries. For code, it uses AST-aware splitting that preserves syntactic units — functions, classes, blocks — rather than cutting mid-expression. A chunk that contains half a function is useless for retrieval and misleading for an LLM that reads it.

Chunks also carry hierarchical context metadata. If a chunk comes from `docs/api/authentication.md`, the system can attach context describing what the `docs/api/` directory contains. This context travels with the chunk when it's returned as a search result, giving the consuming LLM orientation beyond the raw text.

## Query Expansion: Using the LLM Before Retrieval

A user types "how does auth work." BM25 searches for those literal words. Vector search embeds that short phrase, which produces a vague embedding because short queries carry less information than long documents.

QMD optionally runs the query through a local LLM to generate expanded variants. "How does auth work" becomes multiple sub-queries: a lexical variant for BM25 ("authentication flow login token session"), a semantic variant for vector search ("user identity verification and access control"), and a HYDE variant — a hypothetical document that would answer the question, which produces a richer embedding.

```
Original:  "how does auth work"
     │
     ├──→ Lexical:   "authentication flow login token session"
     ├──→ Semantic:   "user identity verification and access control"
     └──→ HYDE:       "Authentication is handled by a JWT-based flow where
                       the client sends credentials to /api/login..."

Each variant is routed to BM25, vector, or both.
```

This is a neat trick. The LLM's knowledge of what authentication *usually* involves generates better search queries than the user's terse input. The cost is one local LLM call before retrieval — slow for interactive search, but fine for agentic workflows where the LLM is already in the loop.

## Built for Agents, Not Humans

QMD exposes its search through three interfaces: a CLI that outputs JSON, a programmatic SDK, and an **MCP server**. The MCP (Model Context Protocol) integration is the one that matters.

MCP is the emerging standard for giving LLM agents access to tools and data sources. An agent connected to QMD's MCP server can search your notes, retrieve specific documents by path, batch-fetch multiple files, and check index status — all through structured tool calls. The agent decides when to search, what to search for, and how to use the results. QMD handles the retrieval.

This is the design choice that separates QMD from a personal search engine. A human searching their notes can tolerate imprecise results — they skim, requery, and recognize what they need. An agent operating autonomously needs high-precision retrieval on the first try, because it will take the top results at face value and reason from them. The three-stage pipeline (retrieve broadly, fuse, rerank precisely) is built for that use case.

## What This Doesn't Solve

**Cold start.** First run requires embedding your entire corpus. For large knowledge bases, this takes real time — embedding models are not fast, even quantized. Incremental updates after that are cheap (only modified files get re-embedded), but the initial indexing is a wall.

**Model quality ceiling.** Everything runs locally on quantized GGUF models. A quantized EmbeddingGemma produces worse embeddings than a full-precision model running on a cluster. A local Qwen3 reranker is less capable than a large cloud reranker. The privacy and latency benefits are real, but there's a quality trade-off.

**Chunk boundary problems.** Even AST-aware chunking can't solve the fundamental issue: relevant information sometimes spans multiple chunks. A question about a function's behavior might need the function definition in one chunk and the calling code in another. No chunking strategy eliminates this without making chunks so large they defeat the purpose of retrieval.

**Semantic drift in long collections.** Vector embeddings encode meaning at a point in time. If your notes use "auth" to mean OAuth in 2024 and API keys in 2026, the embeddings don't distinguish. BM25 has the same problem with polysemy. The reranker partially compensates, but it's working with the same ambiguous text.

## Why It Matters

The direction of LLM tooling is clear: agents that operate on your local data, using your local compute, without phoning home. QMD is a clean implementation of that direction for retrieval. Three-stage hybrid search is a known architecture in information retrieval research, but packaging it as a local-first tool with MCP integration and GGUF models is new.

The broader pattern: the infrastructure layer for LLM agents is being built right now. Context protocol standards, local model runtimes, hybrid retrieval pipelines. The models get the attention. The plumbing determines whether they're useful.

---

**References**

[1] Robertson, S. & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333–389.

[2] Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods. *SIGIR 2009.*

[3] Gao, L., Ma, X., Lin, J., & Callan, J. (2022). Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE). [arXiv:2212.10496](https://arxiv.org/abs/2212.10496)

[4] Anthropic. (2024). Model Context Protocol Specification. [modelcontextprotocol.io](https://modelcontextprotocol.io)
