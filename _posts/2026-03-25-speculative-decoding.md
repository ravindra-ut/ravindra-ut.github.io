---
layout: post
title: "Speculative Decoding: The Free Lunch of LLM Inference"
date: 2026-03-25
categories: [AI, Systems, Performance]
tags: [llm, inference, speculative-decoding, optimization, transformers]
---

*How a small model and a clever acceptance rule can make large language models 2–5x faster, with zero quality loss.*

## The Problem

Large language models generate tokens one at a time. Each token needs a full forward pass: load billions of weights from memory, multiply, sample, repeat. Modern GPUs are compute monsters but memory bottlenecks. An H100 can do 990 TFLOPS but reads memory at only 3.35 TB/s.

When generating one token at a time, the GPU loads all parameters, does a tiny multiply for a single token, then loads everything again. This is **memory-bandwidth bound**: the GPU spends most of its time waiting for data, not doing math.

```
┌─────────────────────────────────────────────────────────────────────┐
│              Standard Autoregressive Decoding                       │
│                                                                     │
│  ┌─────┐  →  ┌─────┐  →  ┌─────┐  →  ┌─────┐  →  ┌─────┐            │
│  │ The │     │ cat │     │ sat │     │  on │     │ the │            │
│  └─────┘     └─────┘     └─────┘     └─────┘     └─────┘            │
│   ~100ms      ~100ms      ~100ms      ~100ms      ~100ms            │
│                                                                     │
│  5 tokens = 5 serial forward passes = ~500ms                        │
└─────────────────────────────────────────────────────────────────────┘
```

*Each token requires loading all model weights from GPU memory — a full forward pass for one token.*

## The Key Insight: Verification Is Parallel

A transformer can **verify** whether it agrees with a sequence of tokens much faster than it can **generate** that sequence from scratch [1].

Why? Verification processes all positions in parallel — attention sees everything at once in a single forward pass. Generation requires a separate forward pass per token.

Use a cheap model to guess, then verify all guesses at once with the expensive model.

## How It Works

The algorithm has two players:

- **Draft model** — a small, fast model (~1B parameters) that guesses tokens quickly.
- **Target model** — the large, expensive model (~70B) whose output quality you care about.

### Step 1: Draft

The small model autoregressively generates K candidate tokens (typically 4–8). This is fast because the draft model is tiny — maybe ~30ms for 5 tokens.

```
Context: "The cat"

Draft model generates:
  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
  │ sat │ │  on │ │ the │ │ mat │ │  .  │    ← ~30ms total
  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘
```

### Step 2: Verify

Feed the original prompt plus all K draft tokens into the large model in **a single forward pass**. The large model computes its own probability distribution at every position simultaneously. This takes roughly the same time as generating one token (~100ms), but evaluates all K positions at once.

```
Large model verifies all 5 drafts in ONE pass:

  "The cat"  +  [sat]  [on]  [the]  [mat]  [.]
                  ↓      ↓      ↓      ↓     ↓       ← ~100ms total
                P=0.85  P=0.92 P=0.88 P=0.15 P=0.30
```

### Step 3: Accept or Reject

Walk through the draft tokens left to right. For each one, compare the draft model's probability with the large model's probability:

```
┌────────────────────────────────────────────────────────────────────┐
│                     The Acceptance Rule                            │
│                                                                    │
│  For each draft token:                                             │
│                                                                    │
│  If P_large(token) ≥ P_draft(token)  →  ALWAYS ACCEPT              │
│     (large model likes it at least as much)                        │
│                                                                    │
│  If P_large(token) < P_draft(token)  →  Accept with probability    │
│     P_large / P_draft, otherwise REJECT and RESAMPLE               │
│                                                                    │
│  Formula:  accept_prob = min(1, P_large / P_draft)                 │
└────────────────────────────────────────────────────────────────────┘
```

At the first rejection, stop, take the resampled token from the large model's distribution, and go back to Step 1.

```
Results:
  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
  │ sat │ │  on │ │ the │ │ mat │ │ rug │
  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘
     ✓        ✓       ✓       ✗       ↻
   accept   accept  accept  reject  resample

  4 tokens accepted + 1 resampled = 5 tokens
  Cost: ~30ms (draft) + ~100ms (verify) = ~130ms
  vs. standard: ~500ms
```

## Why It's Lossless

The acceptance rule is designed so that the final token distribution is **exactly identical** to what you'd get from running the large model alone. This is provably lossless [1][2]. You get a pure speedup without trading quality.

### Acceptance Probability Examples

| P_draft | P_large | min(1, P_large/P_draft) | Result |
|---------|---------|-------------------------|--------|
| 70%     | 85%     | 1.0 (121%)              | Always accept |
| 50%     | 50%     | 1.0 (100%)              | Always accept |
| 80%     | 40%     | 0.50 (50%)              | Accept 50% of the time |
| 90%     | 20%     | 0.22 (22%)              | Accept 22% of the time |

When the large model is *more* confident than the draft, always accept. When the large model is *less* confident, sometimes reject and resample from the large model.

## Speedup in Practice

The speedup depends on the **acceptance rate** — how often the draft model matches the target. For typical English text, a well-matched draft model agrees with the target 70–90% of the time on predictable tokens (articles, prepositions, common continuations) [2].

```
Expected Speedup vs Acceptance Rate (K=5 draft tokens)

  50% acceptance  ████████░░░░░░░░░░░░  1.9x
  60% acceptance  ██████████░░░░░░░░░░  2.3x
  70% acceptance  ████████████░░░░░░░░  2.8x
  80% acceptance  ██████████████░░░░░░  3.4x
  90% acceptance  █████████████████░░░  4.1x
  95% acceptance  ██████████████████░░  4.5x
```

In practice, well-matched draft models yield **2–3x latency improvements**, and up to **4–5x** when the draft model closely approximates the target's distribution. The key variables:

- **Acceptance rate** — how often the draft matches the target (higher = better)
- **Draft model speed** — faster draft = less overhead per round
- **K (speculation length)** — more drafts = higher potential gain, but also higher chance of early rejection

## Variants

### Standard Speculative Decoding
Separate small draft model generates candidates, large model verifies. The original formulation [1][2].

### Self-Speculative Decoding
No separate draft model. Use a cheaper version of the target itself — skip layers, use smaller attention patterns, or early-exit from intermediate layers. Reduces deployment to a single model [3].

### Medusa
Add multiple lightweight prediction heads to the target model, each guessing a different future position (token +2, +3, +4) simultaneously. No separate draft model needed [4].

### Tree-Based Speculation (SpecInfer)
Instead of a single chain of draft tokens, generate a **tree** of candidate continuations at each position. Verify the whole tree at once. If one branch is rejected, another might survive [5].

```
                      ┌── "mat"  ── "."
            ┌── "the" ┤
            │         └── "rug"  ── "."
  "sat" ── "on"
            │         ┌── "a"   ── "warm"
            └── "the" ┤
                      └── "his" ── "lap"

  Tree-based: verify all branches in one pass
  If "mat" is rejected, "rug" branch may survive
```

## Pseudocode

A simplified Python implementation of the core algorithm:

```python
import torch

def speculative_decode(
    target_model,    # Large model (e.g. 70B)
    draft_model,     # Small model (e.g. 1B)
    prompt_tokens,   # Initial context token IDs
    K=5,             # Number of draft tokens per round
    max_tokens=100,  # Total tokens to generate
):
    """
    Speculative decoding: use a small draft model to guess tokens,
    then verify all guesses in one pass with the large target model.
    Output distribution is identical to running target_model alone.
    """
    generated = list(prompt_tokens)

    while len(generated) - len(prompt_tokens) < max_tokens:

        # ── Step 1: Draft ──────────────────────────────────────
        # Small model generates K candidate tokens autoregressively.
        draft_tokens = []
        draft_probs = []
        draft_input = list(generated)

        for _ in range(K):
            logits = draft_model(torch.tensor([draft_input]))
            p = torch.softmax(logits[:, -1, :], dim=-1)
            token = torch.multinomial(p, 1).item()
            draft_tokens.append(token)
            draft_probs.append(p[0, token].item())
            draft_input.append(token)

        # ── Step 2: Verify ─────────────────────────────────────
        # Feed context + all K draft tokens to the large model in ONE pass.
        verify_input = generated + draft_tokens
        logits = target_model(torch.tensor([verify_input]))
        target_probs_all = torch.softmax(logits[:, :, :], dim=-1)

        # ── Step 3: Accept or Reject ──────────────────────────
        # Walk left to right. Accept each token with probability
        # min(1, P_large / P_draft). Stop at first rejection.
        n_accepted = 0

        for i in range(K):
            pos = len(generated) - 1 + i
            p_large = target_probs_all[0, pos, draft_tokens[i]].item()
            p_draft = draft_probs[i]

            accept_prob = min(1.0, p_large / p_draft)

            if torch.rand(1).item() < accept_prob:
                generated.append(draft_tokens[i])
                n_accepted += 1
            else:
                # Resample from adjusted distribution:
                # P_adjusted(x) = max(0, P_large(x) - P_draft(x))
                p_target = target_probs_all[0, pos, :]
                p_draft_full = torch.softmax(
                    draft_model(torch.tensor([generated + draft_tokens[:i]]))[:, -1, :],
                    dim=-1
                )[0]
                adjusted = torch.clamp(p_target - p_draft_full, min=0)
                adjusted = adjusted / adjusted.sum()
                new_token = torch.multinomial(adjusted, 1).item()
                generated.append(new_token)
                break

        # If ALL K tokens accepted, bonus token from target model
        if n_accepted == K:
            bonus_pos = len(generated) - 1
            p_bonus = target_probs_all[0, bonus_pos, :]
            bonus_token = torch.multinomial(p_bonus, 1).item()
            generated.append(bonus_token)

    return generated[len(prompt_tokens):]


# ── Usage ──────────────────────────────────────────────────────
# output = speculative_decode(
#     target_model=llama_70b,
#     draft_model=llama_1b,
#     prompt_tokens=tokenizer.encode("The cat"),
#     K=5,
#     max_tokens=200,
# )
# print(tokenizer.decode(output))
```

## Why This Matters

Training a frontier model is a one-time (massive) cost. Inference — running that model for every query, every API call, every day — is the recurring cost that dwarfs training over time.

A 3x speedup on inference means 3x more users on the same hardware, or GPU costs cut by two-thirds. Combined with quantization and better batching strategies, speculative decoding is part of a set of optimizations making it practical to run large models on less hardware. The race to make inference cheap and fast determines whether these models reach broad deployment.

---

## References

1. Leviathan, Y., Kalman, M., & Matias, Y. (2023). *Fast Inference from Transformers via Speculative Decoding.* ICML 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)

2. Chen, C., Borgeaud, S., Irving, G., et al. (2023). *Accelerating Large Language Model Decoding with Speculative Sampling.* [arXiv:2302.01318](https://arxiv.org/abs/2302.01318) — DeepMind's independent discovery of the same technique.

3. Zhang, J., Wang, J., Li, H., et al. (2023). *Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding.* [arXiv:2309.08168](https://arxiv.org/abs/2309.08168)

4. Cai, T., Li, Y., Geng, Z., Peng, H., & Dao, T. (2024). *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)

5. Miao, X., Oliaro, G., Zhang, Z., et al. (2024). *SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification.* ASPLOS 2024. [arXiv:2305.09781](https://arxiv.org/abs/2305.09781)

6. Stern, M., Shazeer, N., & Uszkoreit, J. (2018). *Blockwise Parallel Decoding for Deep Autoregressive Models.* NeurIPS 2018. [arXiv:1811.03115](https://arxiv.org/abs/1811.03115) — Early precursor to speculative decoding.
