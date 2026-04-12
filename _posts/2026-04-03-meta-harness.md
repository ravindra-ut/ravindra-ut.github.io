---
layout: post
title: "Meta-Harness: When the Scaffolding Optimizes Itself"
date: 2026-04-03
categories: [AI, Agents, Systems]
tags: [llm, harness-engineering, program-synthesis, meta-optimization, agents]
---

*How giving an optimizer a filesystem full of execution traces lets it rewrite the code around a model, not just the prompts.*

## The 6x Gap Nobody Talks About

Change the model powering an LLM system and you might see a 20% improvement. Change the *harness*, the code that decides what the model sees and how it acts, and you can see a 6x swing on the same benchmark with the same model.

This is not surprising once you look at what a harness actually does. In a browser agent, the harness filters the DOM, tracks state changes, structures reflections, and injects observations. In a retrieval-augmented math solver, the harness decides which examples to retrieve and how to format them. In a coding agent, the harness manages context windows, bootstraps the environment, and decides when to summarize. These are engineering decisions that shape every token the model processes, and they compound across every step of a multi-step task.

Yet almost all optimization research focuses on the model: better weights, better prompts, better decoding. The harness is treated as fixed infrastructure. Meta-Harness [1] flips this. It takes the harness itself as the search variable and runs an automated outer loop to find better ones.

## Why Prior Optimizers Fall Short

There is no shortage of methods for optimizing text artifacts with LLMs. OPRO searches over prompts using past (solution, score) pairs. TextGrad computes "textual gradients" on the current artifact. AlphaEvolve maintains a program database with evaluation scores. These methods work, but they share a structural limitation: they compress their feedback into a narrow channel.

The numbers make this concrete:

| Method | Feedback Type | Context Per Iteration |
|--------|--------------|----------------------|
| OPRO | Past scores + solutions | ~2K tokens |
| TextGrad | Textual gradient on current artifact | ~15K tokens |
| AlphaEvolve | Program database + scores | ~22K tokens |
| TTT-Discover | Prior solution fragment | ~26K tokens |
| **Meta-Harness** | **All source code, scores, and execution traces** | **~10M tokens** |

A single harness evaluation can produce millions of tokens of diagnostic information: the full prompts sent to the model, every tool call and its output, every state transition, every error message. Prior methods see at most 26K tokens of this. They know *that* a harness scored 34%; they don't know *why*. Was it a retrieval failure? A formatting issue? A context overflow at step 12 that cascaded into a wrong answer?

Meta-Harness closes this gap by giving the optimizer access to the raw evidence.

## The Core Idea: A Filesystem as Feedback

The architecture is simple. There is a population of candidate harnesses being evaluated on a set of tasks. There is a *proposer*, an agentic coding system (Claude Code in the paper), whose job is to read the results and propose new candidates. And there is a filesystem that stores everything.

For every evaluated harness, the filesystem contains:
1. The full source code
2. Evaluation scores (per-task and aggregate)
3. Complete execution traces: prompts, model outputs, tool calls, state updates, error logs

The proposer does not receive a summary of these results. It receives a filesystem path and standard developer tools: `grep`, `cat`, `ls`, `find`. It navigates the traces the way a developer would debug a failing system, reading specific log files, comparing two candidates' behavior on the same task, searching for patterns across failures.

In practice, the proposer reads a median of 82 files per iteration on the TerminalBench-2 benchmark: 41% harness source code, 40% execution traces, 6% score summaries, 13% other files. This is selective reading, not brute-force ingestion. The proposer decides what to look at based on what it finds.

## The Algorithm

The outer loop is an evolutionary search with no fixed mutation operators:

```
Input: tasks 𝒳, model M, proposer P, iterations N
Initialize population ℋ with baseline harnesses
Initialize filesystem 𝒟 ← ∅

# Evaluate initial population
for H ∈ ℋ:
    E_H ← Evaluate(H, M, 𝒳)
    𝒟 ← 𝒟 ∪ {(H, E_H)}       # store code + scores + traces

# Evolution loop
for t = 1 … N:
    P reads filesystem 𝒟        # grep, cat, diff — whatever it needs
    P proposes k new harnesses {H₁, …, Hₖ}
    for H in {H₁, …, Hₖ}:
        if H passes interface validation:
            𝒟 ← 𝒟 ∪ {(H, Evaluate(H, M, 𝒳))}

return Pareto frontier from 𝒟
```

Two things to notice. First, there is no parent-selection rule. The proposer is free to inspect any prior candidate and its traces when designing a new one. It might base a new harness on the top scorer, or it might notice that a low-scoring candidate had an interesting idea that failed for a fixable reason. Second, each harness is a complete program, not a prompt template or a configuration. The proposer can modify retrieval logic, rewrite context management, add entirely new components, or restructure the control flow. The search space is program space.

## What the Proposer Actually Does

The paper includes a detailed trace of the proposer's behavior on TerminalBench-2, and it reads like a debugging session, not a search algorithm.

**Iterations 1-2:** The proposer bundles two changes: structural fixes to the agent loop and prompt modifications. Both candidates regress from the 64.4% baseline.

**Iteration 3:** The proposer reads the execution traces of the failed candidates and identifies a confound. The regressions were caused by the prompt changes, not the structural fixes. It proposes isolating the structural changes from the prompt.

**Iterations 4-6:** The proposer tests variations on the structural fix alone. All regress. It now has evidence that modifying the agent's completion flow is high-risk, independent of prompt changes.

**Iteration 7:** Strategy shift. Instead of modifying the control loop, the proposer adds an *environment bootstrapping* step that runs before the agent loop starts. It gathers a snapshot of the sandbox (working directory, installed languages, package managers, file listing) and injects it into the initial prompt. The proposer's reasoning: this is "purely additive" and should "eliminate 3-5 wasted exploration turns on dependency-heavy tasks without risking regression."

This candidate achieves the best score. The key insight, that additive information injection is safer than control-flow modification, emerged from reading the traces of failed candidates. No summary or scalar score would have surfaced it.

## Why Raw Traces Beat Summaries

The paper includes an ablation that makes this point sharply:

| Feedback Condition | Median Accuracy | Best Accuracy |
|-------------------|----------------|---------------|
| Scores only | 34.6 | 41.3 |
| Scores + LLM summary | 34.9 | 38.7 |
| Full traces (Meta-Harness) | 50.0 | 56.7 |

Summaries do not just fail to help. They can actively hurt, presumably by compressing away the diagnostically useful details. When the proposer sees that a harness scored 34% on a classification task, it can hypothesize many causes. When it reads the actual prompts the harness constructed and sees that the model received 200K tokens of context with examples sorted randomly rather than by relevance, it knows exactly what to fix.

This matches a pattern familiar to anyone who has debugged a complex system. The bug is rarely in the summary. It is in line 847 of the trace log.

## Three Domains, Three Discovered Harnesses

### Online Text Classification

**Setup:** An LLM receives labeled examples one at a time, updates its memory, and classifies new inputs. The search variable is the harness code that decides how to store, retrieve, and present examples.

**What Meta-Harness found:** A "Label-Primed Query" harness that retrieves examples using TF-IDF similarity, presents them as contrastive pairs anchored to the query, and prepends a coverage block listing all known labels. This achieves 48.6% accuracy versus ACE's 40.9%, a 7.7-point improvement, while using 4x fewer context tokens (45K vs 203K).

The discovered harness requires no additional LLM calls beyond the main task-solving call. All the intelligence is in the retrieval and formatting logic.

### Retrieval-Augmented Math Reasoning

**Setup:** Given a corpus of over 500K solved math problems, retrieve useful examples before solving a new problem. The search variable is the retrieval program.

**What Meta-Harness found:** A four-route BM25 retrieval program that classifies problems into combinatorics, geometry, number theory, or default using lightweight lexical predicates, then applies route-specific retrieval and formatting. A single harness, discovered on 250 search problems, improves accuracy by an average of 4.7 points across five held-out models on 200 IMO-level problems.

The transfer result is the interesting part. The retrieval harness was optimized with one model but improved all five, including models from different providers. The harness captures task structure, not model-specific quirks.

### Agentic Coding (TerminalBench-2)

**Setup:** 89 Dockerized tasks spanning code translation, distributed ML, systems programming, bioinformatics, and cryptanalysis. Binary pass/fail grading, 5 independent trials per task, fully autonomous execution.

**What Meta-Harness found:** The environment bootstrapping harness described earlier. Starting from Terminus-KIRA (74.7%), Meta-Harness reaches 76.4% on Claude Opus 4.6, ranking #2 among all agents. On Claude Haiku 4.5, it reaches 37.6%, ranking #1 among all Haiku agents.

The discovered improvement is strikingly simple: a single compound shell command run before the agent loop that gathers environment metadata. The value is not in the idea (which any engineer might have) but in the *evidence-driven process* that identified it as the highest-leverage change after systematically ruling out alternatives.

## The Efficiency Argument

Meta-Harness uses far more compute per iteration than prior methods (10M tokens versus ~20K). But it uses far fewer iterations. On text classification, it matches the final accuracy of OpenEvolve and TTT-Discover in 4 evaluations versus their 60+, then surpasses them by over 10 points.

The economics favor Meta-Harness when evaluations are expensive. Running a coding agent on 89 Docker containers 5 times each is not cheap. Neither is classifying thousands of examples across three datasets. When each evaluation costs real money and wall-clock time, spending more on diagnosis to waste fewer evaluations is a good trade.

## What This Changes

The traditional pipeline for building LLM systems is: choose a model, write a harness, tune the prompts, ship it. The harness is designed once by an engineer and rarely revisited. Meta-Harness suggests a different pipeline: write a baseline harness, define an evaluation, and let the optimization loop find a better harness.

This does not eliminate harness engineering. Someone still needs to define the search space, write the evaluation, and provide reasonable starting points. But it changes what the engineer optimizes. Instead of manually tuning retrieval logic and prompt formatting, the engineer designs the evaluation that guides automated search. The bottleneck shifts from *implementation* to *specification*.

There is a deeper implication. If harness code is a first-class optimization target, then the boundary between "the model" and "the system around it" becomes less meaningful. What matters is the end-to-end performance of the system on the task distribution. Whether a capability comes from model weights, prompt engineering, or harness logic is an implementation detail. Meta-Harness optimizes the part of the system that is cheapest to change, program code, while leaving the most expensive part, model weights, fixed.

---

**References**

[1] Lee, Y., Nair, R., Zhang, Q., Lee, K., Khattab, O., Finn, C. "Meta-Harness: End-to-End Optimization of Model Harnesses." *Preprint, 2026.* [arXiv:2603.28052](https://arxiv.org/abs/2603.28052)

[2] Fernando, C., et al. "AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery." *2025.*

[3] Khattab, O., et al. "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines." *ICLR 2024.*

[4] Yang, C., et al. "Large Language Models as Optimizers." *ICLR 2024.*
