# Speculative Decoding: The Free Lunch of LLM Inference

*How a small model and a clever acceptance rule can make large language models 2–5x faster — with zero quality loss.*

---

## The Problem: GPUs Hate Generating Text

Large language models generate tokens one at a time. Each token needs a full forward pass: load billions of weights from memory, multiply, sample, repeat. Modern GPUs are compute monsters but memory bottlenecks — an H100 can do 990 TFLOPS but reads memory at only 3.35 TB/s.

When generating one token at a time, the GPU loads all parameters, does a tiny multiply for a single token, then loads everything again. This is called being **memory-bandwidth bound**. The GPU spends most of its time waiting for data, not doing math.

```
┌─────────────────────────────────────────────────────────────────────┐
│              Standard Autoregressive Decoding                       │
│                                                                     │
│  ┌─────┐  →  ┌─────┐  →  ┌─────┐  →  ┌─────┐  →  ┌─────┐        │
│  │ The │     │ cat │     │ sat │     │  on │     │ the │        │
│  └─────┘     └─────┘     └─────┘     └─────┘     └─────┘        │
│   ~100ms      ~100ms      ~100ms      ~100ms      ~100ms          │
│                                                                     │
│  5 tokens = 5 serial forward passes = ~500ms                       │
└─────────────────────────────────────────────────────────────────────┘
```

*Each token requires loading all model weights from GPU memory — a full forward pass just for one token.*

---

## The Key Insight: Verification Is Parallel

Here's the observation that makes speculative decoding work: a transformer can **verify** whether it agrees with a sequence of tokens much faster than it can **generate** that sequence from scratch [1].

Why? Verification processes all positions in parallel — attention sees everything at once in a single forward pass. But generating those same tokens sequentially requires separate forward passes.

This asymmetry is the entire foundation. Use a cheap model to guess, then verify all guesses at once with the expensive model.

---

## How It Works

The algorithm has two players:

- **Draft model** — a small, fast model (~1B parameters) that guesses tokens quickly
- **Target model** — the large, expensive model (~70B) whose output quality you care about

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
│  If P_large(token) ≥ P_draft(token)  →  ALWAYS ACCEPT             │
│     (large model likes it at least as much)                        │
│                                                                    │
│  If P_large(token) < P_draft(token)  →  Accept with probability   │
│     P_large / P_draft, otherwise REJECT and RESAMPLE               │
│                                                                    │
│  Formula:  accept_prob = min(1, P_large / P_draft)                 │
└────────────────────────────────────────────────────────────────────┘
```

The moment you hit a rejection, stop, take the resampled token from the large model's distribution, and go back to Step 1.

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

---

## Why It's Lossless

The acceptance scheme isn't arbitrary — it's designed so that the final token distribution is **exactly identical** to what you'd get from running the large model alone. This is provably lossless. You're not trading quality for speed. It's a pure speedup for free [1][2].

### Acceptance Probability Examples

| P_draft | P_large | min(1, P_large/P_draft) | Result |
|---------|---------|-------------------------|--------|
| 70%     | 85%     | 1.0 (121%)              | Always accept |
| 50%     | 50%     | 1.0 (100%)              | Always accept |
| 80%     | 40%     | 0.50 (50%)              | Accept 50% of the time |
| 90%     | 20%     | 0.22 (22%)              | Accept 22% of the time |

When the large model is *more* confident than the draft → always accept.
When the large model is *less* confident → sometimes reject, resample from the large model.

---

## Speedup in Practice

The speedup depends primarily on the **acceptance rate** — how often the draft model matches the target. For typical English text, a well-matched draft model agrees with the target 70–90% of the time on predictable tokens (articles, prepositions, common continuations) [2].

```
Expected Speedup vs Acceptance Rate (K=5 draft tokens)

  50% acceptance  ████████░░░░░░░░░░░░  1.9x
  60% acceptance  ██████████░░░░░░░░░░  2.3x
  70% acceptance  ████████████░░░░░░░░  2.8x
  80% acceptance  ██████████████░░░░░░  3.4x
  90% acceptance  █████████████████░░░  4.1x
  95% acceptance  ██████████████████░░  4.5x
```

In practice, teams report **2–3x latency improvements** with well-matched draft models, and up to **4–5x** in favorable conditions. The key variables are:

- **Acceptance rate** — how often draft matches target (higher = better)
- **Draft model speed** — faster draft = less overhead per round
- **K (speculation length)** — more drafts = higher potential gain, but also higher chance of early rejection

---

## Variants and Extensions

The original idea has spawned a family of techniques:

### Standard Speculative Decoding
Separate small draft model generates candidates, large model verifies. The original formulation [1][2].

### Self-Speculative Decoding
Skip the separate draft model entirely. Use a cheaper version of the target itself — skip layers, use smaller attention patterns, or early-exit from intermediate layers. Simplifies deployment to a single model [3].

### Medusa
Add multiple lightweight prediction heads to the target model, each guessing a different future position (token +2, +3, +4) simultaneously. No separate draft model needed — the target grows extra "tentacles" for parallel prediction [4].

### Tree-Based Speculation (SpecInfer)
Instead of a single chain of draft tokens, generate a **tree** of candidate continuations at each position. Verify the whole tree at once. If one branch is rejected, another might still be accepted — hedging against bad guesses [5].

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

---

## Pseudocode

Here's a simplified Python implementation that captures the core algorithm:

```python
import torch

def speculative_decode(
    target_model,    # Large model (e.g. 70B) — the one we want output from
    draft_model,     # Small model (e.g. 1B) — the cheap guesser
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
        # This is fast because the draft model is tiny.
        draft_tokens = []
        draft_probs = []
        draft_input = list(generated)

        for _ in range(K):
            logits = draft_model(torch.tensor([draft_input]))
            p = torch.softmax(logits[:, -1, :], dim=-1)  # distribution over vocab
            token = torch.multinomial(p, 1).item()        # sample next token
            draft_tokens.append(token)
            draft_probs.append(p[0, token].item())         # save P_draft(token)
            draft_input.append(token)

        # ── Step 2: Verify ─────────────────────────────────────
        # Feed context + all K draft tokens to the large model in ONE pass.
        # The large model computes its probability at each position in parallel.
        verify_input = generated + draft_tokens
        logits = target_model(torch.tensor([verify_input]))

        # Extract P_large at each draft position
        # Position i in draft corresponds to position len(generated)-1+i in the sequence
        target_probs_all = torch.softmax(logits[:, :, :], dim=-1)

        # ── Step 3: Accept or Reject ──────────────────────────
        # Walk left to right. Accept each token with probability
        # min(1, P_large / P_draft). Stop at first rejection.
        n_accepted = 0

        for i in range(K):
            pos = len(generated) - 1 + i  # position in the full sequence
            p_large = target_probs_all[0, pos, draft_tokens[i]].item()
            p_draft = draft_probs[i]

            # The core acceptance rule
            accept_prob = min(1.0, p_large / p_draft)

            if torch.rand(1).item() < accept_prob:
                # ✓ Accepted — this token matches the target distribution
                generated.append(draft_tokens[i])
                n_accepted += 1
            else:
                # ✗ Rejected — resample from adjusted distribution:
                # P_adjusted(x) = max(0, P_large(x) - P_draft(x))
                # This correction ensures exact target distribution
                p_target = target_probs_all[0, pos, :]
                p_draft_full = torch.softmax(
                    draft_model(torch.tensor([generated + draft_tokens[:i]]))[:, -1, :],
                    dim=-1
                )[0]
                adjusted = torch.clamp(p_target - p_draft_full, min=0)
                adjusted = adjusted / adjusted.sum()  # normalize
                new_token = torch.multinomial(adjusted, 1).item()
                generated.append(new_token)
                break  # stop checking further drafts

        # If ALL K tokens accepted, we also get a bonus token
        # from the target model's prediction at position K
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

**Key things to notice in the code:**

- **Step 1** is the only sequential part — the draft model generates K tokens one by one, but it's tiny so this is fast (~30ms).
- **Step 2** is a single forward pass through the large model with all K candidates. This is where the parallelism wins — one pass costs ~100ms regardless of K.
- **Step 3** walks left to right with the `min(1, P_large / P_draft)` rule. The `adjusted` distribution on rejection ensures mathematical equivalence to the target model.
- **Bonus token**: if all K drafts are accepted, the target model's output at position K gives us one extra token for free.

---

## Why This Matters

Training a frontier model is a one-time (massive) cost. Inference — running that model for every query, every API call, every day — is the recurring bill that dwarfs training over time.

> "Training is the flex. Inference is the forever bill."

A 3x speedup on inference means 3x more users on the same hardware, or GPU costs cut by two-thirds. Combined with quantization techniques (like compressing KV-cache precision) and better batching strategies, speculative decoding is part of a wave of optimizations making it possible to run serious LLM workloads on surprisingly modest hardware.

The race to train the biggest model gets the headlines. The race to make inference cheap and fast? That's what determines whether these models actually reach everyone.

---

## TL;DR

Speculative decoding uses a small, fast model to guess several tokens ahead, then verifies all guesses at once with the large model. Accepted guesses are free tokens. Rejected guesses get resampled. The output is mathematically identical to running the large model alone — you just get there faster. Typical speedup: **2–5x, zero quality loss**.

---

## References

1. Leviathan, Y., Kalman, M., & Matias, Y. (2023). *Fast Inference from Transformers via Speculative Decoding.* ICML 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)

2. Chen, C., Borgeaud, S., Irving, G., et al. (2023). *Accelerating Large Language Model Decoding with Speculative Sampling.* [arXiv:2302.01318](https://arxiv.org/abs/2302.01318) — DeepMind's independent discovery of the same technique.

3. Zhang, J., Wang, J., Li, H., et al. (2023). *Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding.* [arXiv:2309.08168](https://arxiv.org/abs/2309.08168)

4. Cai, T., Li, Y., Geng, Z., Peng, H., & Dao, T. (2024). *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)

5. Miao, X., Oliaro, G., Zhang, Z., et al. (2024). *SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification.* ASPLOS 2024. [arXiv:2305.09781](https://arxiv.org/abs/2305.09781)

6. Stern, M., Shazeer, N., & Uszkoreit, J. (2018). *Blockwise Parallel Decoding for Deep Autoregressive Models.* NeurIPS 2018. [arXiv:1811.03115](https://arxiv.org/abs/1811.03115) — Early precursor to speculative decoding.
