---
layout: post
title: "Memento: Teaching LLM Agents to Learn Without Touching the Weights"
date: 2026-04-12
categories: [AI, Agents, RL]
tags: [llm, agents, memory, case-based-reasoning, reinforcement-learning, mdp]
---

*How a case bank and a lightweight retrieval policy let an LLM agent improve from experience — no gradient updates to the model required.*

## The Problem with Agent Learning

LLM agents fail, learn nothing from failing, and fail the same way next time.

A coding agent hits a Docker networking issue, burns 30 steps debugging it, eventually finds the fix. Tomorrow it encounters the same issue on a different task. It burns 30 steps again. The model has no mechanism to carry forward what worked. Every task starts from the same blank slate.

The standard fix is fine-tuning. Collect trajectories, compute gradients, update model weights. This works — DeepSeek-R1, OpenAI's o-series models, and others demonstrate that RL over trajectories improves agent behavior. But fine-tuning an LLM is expensive. It requires GPU clusters, careful hyperparameter tuning, catastrophic-forgetting mitigation, and a pipeline that most teams cannot maintain. It also bakes learned behavior into the weights permanently. An agent that learned Docker debugging patterns through fine-tuning carries those patterns into every future task, even when they're irrelevant.

Reflection-based agents (Reflexion, LATS) take a lighter approach: after failure, ask the model to reflect on what went wrong and carry that reflection forward as context. But reflection is static text. The model generates a natural language summary of its experience, and that summary is appended to future prompts. There's no mechanism to retrieve the *right* reflection for a *new* task, no way to update a reflection that turns out to be wrong, and no principled scoring of which reflections actually helped.

[Memento](https://arxiv.org/abs/2508.16153) (Zhou et al., 2025) takes a different approach. Instead of updating model weights or generating static reflections, it stores past experiences as structured cases in a memory bank and learns to retrieve the right ones. The model never changes. The memory around it does.

## Case-Based Reasoning: An Old Idea, Rewired

The core mechanism is case-based reasoning (CBR), a technique from the 1990s AI literature that predates neural networks. The idea: when you encounter a new problem, find the most similar problem you've solved before, and adapt that solution.

In Memento, a **case** is a triplet: `(task, plan, outcome)`. The task is the problem description. The plan is the sequence of subtasks the agent executed. The outcome is binary: did it work or not? After every task attempt, the agent stores a new case. Before every new task, it retrieves the most relevant past cases and uses them to inform its plan.

```
Case Bank (after 50 tasks):

┌─────────────────────────────────────────────────────────────┐
│  Case 1:                                                    │
│    Task: "What is the total mass of the planets in the      │
│           solar system that have rings?"                     │
│    Plan: [search for planets with rings → extract masses    │
│           → compute sum]                                    │
│    Outcome: ✓ Success                                       │
│                                                             │
│  Case 2:                                                    │
│    Task: "Find the GDP per capita of the country that       │
│           hosted the 2022 World Cup"                        │
│    Plan: [search for 2022 World Cup host → search GDP       │
│           data → extract per-capita figure]                 │
│    Outcome: ✗ Failure (search returned outdated GDP)        │
│                                                             │
│  Case 3: ...                                                │
│  Case 4: ...                                                │
│  ...                                                        │
└─────────────────────────────────────────────────────────────┘
```

When a new task arrives — say, "What is the combined area of African countries that border the Mediterranean?" — the agent retrieves similar cases. Case 1 is structurally similar: identify a subset of entities matching a criterion, extract a numeric property, aggregate. The agent sees that the plan `[search for subset → extract property → compute aggregate]` worked before, and uses it as a template.

Critically, the agent also sees Case 2's failure and its reason. A case bank is not just a library of successes. Failed cases teach the agent what *not* to do, or more precisely, they teach the retrieval system to downweight plans that didn't work in similar contexts.

## The Formal Setup: Memory-Augmented MDP

Memento formalizes this as a **Memory-Based Markov Decision Process (M-MDP)**. A standard MDP has states, actions, transitions, and rewards. An M-MDP adds a memory space: the set of all past experience tuples the agent has accumulated.

```
Standard MDP:  ⟨S, A, P, R, γ⟩
M-MDP:         ⟨S, A, P, R, γ, M⟩

where M = (S × A × R)*  — the growing memory of past experiences
```

The agent's policy decomposes into two parts. A **retrieval policy** `μ` selects which cases to pull from memory. A **language model policy** `p_LLM` generates the action (the plan) conditioned on the current task and the retrieved cases:

```
π(action | state, memory) = Σ  μ(case | state, memory) · p_LLM(action | state, case)
                            cases
```

The LLM's parameters are frozen. Learning happens entirely in the retrieval policy `μ`. This is the key insight: instead of adjusting billions of model parameters to improve agent behavior, adjust which experiences the model sees.

## Two Ways to Retrieve: Similarity vs. Learned Value

Memento implements two retrieval strategies. Both maintain the same case bank. They differ in how they decide which cases to surface.

### Non-Parametric: Nearest Neighbors

The simplest approach. Encode the current task and all stored tasks using a frozen text encoder (SimCSE). Retrieve the K cases whose task embeddings are most similar to the current one by cosine similarity.

```
Retrieve(current_task, case_bank) = TopK by cosine_sim(encode(current_task), encode(stored_task))
```

This works the way vector search always works: semantically similar tasks get similar embeddings, and nearby cases get retrieved. A question about "countries bordering the Mediterranean" retrieves cases about geographic filtering because the embeddings are close.

The limitation is also familiar from vector search: similarity is not utility. A case might be semantically close but unhelpful — or semantically distant but structurally analogous. The retriever has no way to learn this distinction.

### Parametric: Learned Q-Function

The more interesting approach. Instead of ranking cases by similarity, rank them by a learned **Q-function** that estimates how useful each case will be for the current task.

The Q-function is a small 2-layer MLP. It takes two inputs: the encoded current task and the encoded stored case. It outputs a score: how likely is this case to lead to success if retrieved?

```
Q(current_task, case) → predicted utility score

Retrieve(current_task, case_bank) = TopK by Q(current_task, case)
```

The Q-function is trained online using soft Q-learning. After every task attempt, the agent observes the outcome (success or failure) and updates the Q-function to reflect whether the retrieved cases actually helped. The training signal is binary cross-entropy — did the case contribute to a successful outcome or not?

```
Loss = -r · log Q(s, c) - (1-r) · log(1 - Q(s, c))
```

This is a tiny model — a 2-layer MLP with a few thousand parameters, trained on individual (task, case, outcome) triples. It updates in milliseconds. The LLM, with its billions of parameters, remains frozen.

The soft Q-learning formulation adds an entropy bonus that prevents the retrieval policy from collapsing to always retrieving the same cases:

```
μ*(case | state, memory) = exp(Q*(state, case) / α) / Σ exp(Q*(state, case') / α)
```

This is just a temperature-scaled softmax over Q-values. High-value cases are retrieved more often, but the entropy term ensures the agent occasionally tries retrieving different cases — exploration in memory space.

## The Architecture: Planner and Executor

Memento uses a two-stage architecture. A **planner** decomposes the task into subtasks. An **executor** carries out each subtask using tools.

```
┌──────────────────────────────────────────────────────────┐
│                     Memento Agent                        │
│                                                          │
│  ┌──────────┐    ┌───────────────┐    ┌──────────────┐  │
│  │ New Task │───→│   Retriever   │───→│   Planner    │  │
│  └──────────┘    │ (Q-function   │    │ (GPT-4.1)    │  │
│                  │  or cosine)   │    │              │  │
│                  └───────┬───────┘    └──────┬───────┘  │
│                          │                   │          │
│                    K cases              Subtask plan    │
│                          │                   │          │
│                          ▼                   ▼          │
│                  ┌───────────────┐    ┌──────────────┐  │
│                  │  Case Bank    │    │  Executor    │  │
│                  │              │    │  (o3/o4-mini) │  │
│                  │  (task,      │    │              │  │
│                  │   plan,      │    │  Tools:      │  │
│                  │   outcome)   │◄───│  - search    │  │
│                  │              │    │  - crawl     │  │
│                  └──────────────┘    │  - code exec │  │
│                        ▲            │  - vision    │  │
│                        │            │  - audio     │  │
│                        │            └──────────────┘  │
│                        │                   │          │
│                  Store new case       Final answer    │
│                  after completion           │          │
│                                            ▼          │
└──────────────────────────────────────────────────────────┘
```

A design choice that matters: the planner is a **fast, non-deliberative** model (GPT-4.1), not a slow reasoning model. The paper tests this directly — using o3 as the planner instead of GPT-4.1 drops average accuracy from 70.9% to 63.0% on GAIA. Slow planners produce verbose, overly detailed plans that mislead the executor. Concise task decomposition works better. The reasoning happens at execution time, not planning time.

The executor has access to a full tool suite via MCP: meta-search (SearxNG aggregating Google, Bing, DuckDuckGo), web crawling, sandboxed Python execution, vision-language models for images, speech recognition for audio, spreadsheet and PDF parsers. The executor runs autonomously for each subtask, selecting tools and iterating until it has a result.

## Results: What the Memory Buys

### GAIA Benchmark

GAIA tests general-purpose AI assistants on multi-step reasoning tasks requiring tool use, web search, and document processing. Three difficulty levels.

| System | Level 1 | Level 2 | Level 3 | Average |
|--------|---------|---------|---------|---------|
| Memento | 96.23% | 90.70% | 61.54% | **87.88%** |
| Alita | — | — | — | 87.27% |
| OpenAI Deep Research | — | — | — | 67.40% |
| Open Deep Research | — | — | — | 55.15% |

87.88% Pass@3 on validation, 79.40% on the test set. Top-1 among open-source frameworks.

### DeepResearcher (7 QA Datasets)

| System | F1 | PM |
|--------|----|----|
| **Memento** | **66.6** | **80.4** |
| DeepResearcher (fine-tuned) | 51.8 | 60.5 |
| Search-r1-base (fine-tuned) | 48.3 | 53.8 |
| CoT + RAG | 37.7 | 43.2 |
| CoT only | 23.6 | 26.1 |

Memento nearly doubles the F1 of CoT+RAG without any model training. It also outperforms DeepResearcher, which *does* fine-tune its underlying model. Memory retrieval beats gradient updates, at a fraction of the compute cost.

### SimpleQA and HLE

95.0% on SimpleQA (new state-of-the-art). 24.4% on Humanity's Last Exam — 0.9 points behind GPT-5, ahead of Gemini 2.5 Pro and o3.

## How Many Cases Do You Need?

The ablation on K — the number of cases retrieved per task — reveals a clean pattern:

```
K=0  (no memory):    F1=59.9   PM=72.2
K=1:                 F1=63.6   PM=77.9    ← big jump from even 1 case
K=2:                 F1=63.7   PM=78.1
K=4:                 F1=64.5   PM=78.5    ← peak
K=8:                 F1=64.1   PM=78.2
K=16:                F1=63.9   PM=78.1
K=32:                F1=63.9   PM=78.1    ← no benefit, slight decline
```

One retrieved case buys most of the improvement. Four is optimal. Beyond that, adding more cases introduces noise without adding signal — the additional cases are less relevant but still consume context tokens. Small, curated memory beats large, noisy memory.

This mirrors what we know about few-shot prompting: a handful of well-chosen examples outperforms a wall of mediocre ones.

## Continual Improvement Without Catastrophic Forgetting

Memento runs iteratively. Each iteration processes a batch of tasks, stores the results as new cases, and starts the next iteration with a richer case bank. Over five iterations on DeepResearcher:

```
Iteration    No CBR    Non-Parametric    Parametric
    1        78.65%       79.84%          80.46%
    2        80.93%       81.87%          82.84%
    3        82.62%       83.09%          84.10%
    4        83.53%       84.03%          84.85%
    5        84.47%       84.85%          85.44%
```

Three things stand out. First, performance improves monotonically across iterations — the agent genuinely learns from accumulated experience. Second, parametric retrieval (the learned Q-function) consistently outperforms non-parametric (cosine similarity) by about 0.5-1 point. Third, even the no-CBR baseline improves, because later iterations benefit from improved executor tools and accumulated subtask history. But CBR adds a consistent margin on top.

No catastrophic forgetting. The model weights never change, so there's nothing to forget. New cases enter the bank alongside old ones. The retrieval policy learns to select from a growing pool.

## Out-of-Distribution Generalization

The strongest result: cases learned on one set of benchmarks transfer to unseen task types. Memento trains its case bank on NQ, TriviaQA, HotpotQA, and 2WikiMultiHopQA. Then it is tested on three held-out datasets it has never seen:

| OOD Dataset | Improvement from CBR |
|-------------|---------------------|
| MusiQue | +4.7% |
| Bamboogle | +9.6% |
| PopQA | +6.1% |

These are not trivial gains. +9.6% absolute on Bamboogle means the agent is solving tasks it couldn't before, on a task distribution it was never exposed to. The case bank captures structural patterns — "decompose multi-hop questions into sequential lookups," "verify intermediate facts before aggregating" — that generalize beyond the specific questions used to build the bank.

## What This Doesn't Solve

**Cold start.** With an empty case bank, Memento is just a planner-executor without memory. The first iteration has no cases to retrieve. Performance gains require accumulated experience, which requires running tasks, which costs API calls.

**Case bank curation.** The bank grows monotonically. There's no forgetting mechanism, no pruning of outdated or misleading cases. On a stationary task distribution this is fine. On a shifting one — where facts change, tools change, or task types evolve — stale cases could degrade retrieval quality.

**Attribution opacity.** When the agent retrieves four cases and produces a plan, it's unclear which case influenced which planning decision. The Q-function scores cases individually, but the LLM processes them as a batch in its context. Debugging why the agent chose a bad plan requires tracing through the interaction between retrieved cases and the planner's reasoning — the same interpretability problem that afflicts all in-context learning.

**Ceiling effects.** Memento's performance is bounded by the executor's capabilities. If the underlying models can't solve a subtask — if o3 can't write the code or GPT-4o can't parse the image — no amount of memory helps. The case bank improves *planning*, not *execution*. On Humanity's Last Exam, where tasks require frontier-level reasoning, Memento's 24.4% sits just below GPT-5's 25.3%. The gap is in the executor, not the memory.

## Why It Matters

The dominant assumption in LLM agent research is that improving agents means improving models — bigger models, better training data, more RLHF. Memento challenges this by showing that a frozen model with a growing memory bank can outperform fine-tuned models on complex benchmarks.

The efficiency argument is stark. Fine-tuning a 70B model on agent trajectories requires hundreds of GPU-hours and careful pipeline engineering. Memento's parametric retriever is a 2-layer MLP that trains in milliseconds on a CPU. The computational cost of "learning" is essentially zero. The only cost is running the tasks themselves.

This connects to a broader pattern in the current LLM systems landscape. [Meta-Harness]({% post_url 2026-04-03-meta-harness %}) optimizes the *code* around a frozen model. QMD optimizes *retrieval* for a frozen model. Memento optimizes *experience selection* for a frozen model. All three treat the model as a fixed resource and invest optimization budget in the infrastructure around it. The model is the most expensive component to change and the cheapest to keep constant. Everything else — harness code, retrieval pipelines, memory systems — is lightweight and updatable.

The CBR framing also offers something that fine-tuning doesn't: **inspectability**. You can look at the case bank. You can see which cases were retrieved for a given task and whether they were successes or failures. You can manually add, remove, or edit cases. The learned behavior is stored in a data structure, not compressed into weight matrices. When the agent makes a bad decision, you can trace it back to the cases it was shown.

An agent that remembers what worked, retrieves it when relevant, and learns which memories to trust — without changing a single parameter of the model it runs on. The model provides the reasoning. The memory provides the experience. The split is clean.

---

**References**

[1] Zhou, H., Chen, Y., Guo, S., et al. (2025). Memento: Fine-tuning LLM Agents without Fine-tuning LLMs. [arXiv:2508.16153](https://arxiv.org/abs/2508.16153)

[2] Shinn, N., Cassano, F., Gopinath, A., et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)

[3] Aamodt, A. & Plaza, E. (1994). Case-Based Reasoning: Foundational Issues, Methodological Variations, and System Approaches. *AI Communications*, 7(1), 39–59.

[4] Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning. *ICML 2018.* [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)

[5] Guo, D., Yang, D., Zhang, H., et al. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
