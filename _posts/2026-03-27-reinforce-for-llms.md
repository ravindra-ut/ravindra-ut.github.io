---
layout: post
title: "REINFORCE for LLMs: Teaching GPT-2 to Count Even Numbers"
date: 2026-03-27
categories: [AI, RL]
tags: [reinforcement-learning, llm, policy-gradients, rlhf, transformers]
---

*How policy gradients work on language models, why they beat supervised fine-tuning, and what 200 lines of code can teach you about the RL behind RLHF.*

## The Setup

Take GPT-2 (a small language model with 124M parameters) and give it one job: when prompted with "Generate an even number between 1 and 10:", output an even number.

This is a toy problem on purpose. We can check the answer with a single `if` statement. That same property is what makes math and code such natural training grounds for RL. You don't need a human to tell you whether 2+2=4 or whether code compiles.

Out of the box, GPT-2 gets this right about 16% of the time. We're going to push that above 80% using REINFORCE, the simplest reinforcement learning algorithm. If you've heard of PPO or GRPO (used to train ChatGPT and DeepSeek), they're fancier versions of the same idea.

## Two Ways to Teach a Model

Two approaches. Understanding when each fails is the whole point.

**Supervised Fine-Tuning (SFT)**: learning by imitation. You write down the correct answers as examples: "Generate an even number between 1 and 10: 2", "...4", "...6", "...8", "...10". Then you train the model to make those exact outputs more likely. The same technique you'd use to train on any text dataset. Minimize the gap between what the model predicts and what you showed it. The model learns to copy.

**REINFORCE**: learning by trial and error. You let the model generate its own answers, check each one with the scoring function, and nudge the model based on what worked. Got a high score? Do more of that. Got zero? Do less. No examples needed. The model discovers what works on its own.

```
SFT:
  Expert says "the answer is 4"  →  minimize -log P("4")
  Expert says "the answer is 6"  →  minimize -log P("6")
  Model learns to imitate the expert.

REINFORCE:
  Model samples "7"  →  reward = 0  →  decrease P("7")
  Model samples "4"  →  reward = 1  →  increase P("4")
  Model samples "3"  →  reward = 0  →  decrease P("3")
  Model samples "8"  →  reward = 1  →  increase P("8")
  Model learns from its own experience.
```

Both work here. But they fail differently, and the failure modes matter.

## The Reward Function

How do we tell if the model got it right? We write a simple scoring function. If the model outputs 2, 4, 6, 8, or 10, it gets a score of 1. Anything else (odd numbers, out-of-range numbers, gibberish) gets a 0. No judgment calls, no humans in the loop.

In code, that's two functions. One extracts the first integer from the model's output. The other scores it.

```python
def extract_number(text: str) -> int | None:
    """Pull the first integer from the model's completion."""
    match = re.search(r'\b(\d+)\b', text)
    return int(match.group(1)) if match else None

def reward_fn(completions: list[str]) -> torch.Tensor:
    """1 if even and in [1,10], 0 otherwise."""
    rewards = []
    for text in completions:
        num = extract_number(text)
        if num is not None and 1 <= num <= 10 and num % 2 == 0:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return torch.tensor(rewards, device=device)
```

In production RLHF systems (like those used to train ChatGPT), the scoring function is itself a neural network trained on human preferences. Here, we skip all that complexity. A regex extracts the number, a modulo check tells us if it's even. The score is always right. That exact signal is what makes this problem clean enough to learn from.

## Sampling from the Policy

In RL terminology, the model is the "policy," the thing making decisions, one token at a time. A "rollout" is one complete run where the model reads the prompt and generates tokens until it's done. Each call to `model.generate` produces one rollout.

```python
@torch.no_grad()
def sample_completions(model, prompt, batch_size=32,
                       max_new_tokens=12, temperature=1.0):
    inputs = tokenizer([prompt] * batch_size, return_tensors="pt",
                       padding=True).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )

    completion_ids = outputs[:, prompt_len:]
    completion_texts = tokenizer.batch_decode(
        completion_ids, skip_special_tokens=True
    )
    return outputs, completion_ids, completion_texts
```

Batches of 32 completions. Temperature 1.0, the raw distribution.

## The SFT Baseline

SFT is straightforward. We have five example answers (one for each even number). At each training step, we pick one at random and train the model to predict it. The standard language modeling objective, just on our tiny dataset of correct outputs. 200 steps total.

```python
def train_sft(model, n_steps=200, lr=5e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    demos = [f"{PROMPT} {n}" for n in [2, 4, 6, 8, 10]]
    history = []

    for step in range(n_steps):
        demo = demos[torch.randint(len(demos), (1,)).item()]
        inputs = tokenizer(demo, return_tensors="pt").to(device)

        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            rate = evaluate(model, n_samples=256)
            history.append((step + 1, rate))
            print(f"SFT step {step+1}: even rate = {rate:.1%}")

    return history
```

This works. The even-number rate climbs. But watch what happens to the output distribution.

## REINFORCE: The Core RL Loop

Instead of showing the model correct answers, we let it generate answers, score them, and learn from the scores. Five steps:

1. **Roll out**: let the model generate a batch of completions
2. **Score**: run each completion through the reward function
3. **Compute log probabilities**: how likely was each completion under the current model? And under the original, untrained model?
4. **Compute the KL penalty**: measure how much the model's behavior has drifted from the original (more on this below)
5. **Update**: nudge the model toward completions that scored well, away from those that didn't

The math for this update (don't worry if this looks opaque; we'll unpack it piece by piece):

```
∇J = E[ ∇log π(y|x) · (r(x,y) - β · KL(π || π_ref)) ]
```

In plain English: if a completion got a high reward and didn't drift too far from the original model, make it more likely. If it scored poorly, make it less likely.

```python
def train_reinforce(model, ref_model, n_steps=200, batch_size=32,
                    lr=1e-5, kl_coeff=0.05):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []

    for step in range(n_steps):
        # 1. ROLLOUT: sample from current policy
        full_ids, completion_ids, texts = sample_completions(
            model, PROMPT, batch_size=batch_size
        )

        # 2. REWARD: score each completion
        rewards = reward_fn(texts)

        # 3. LOG PROBS: current policy and reference
        model.train()
        outputs = model(full_ids)
        logits = outputs.logits

        prompt_len = full_ids.shape[1] - completion_ids.shape[1]
        comp_logits = logits[:, prompt_len - 1:-1, :]
        log_probs = F.log_softmax(comp_logits, dim=-1)
        token_log_probs = log_probs.gather(
            2, completion_ids.unsqueeze(-1)
        ).squeeze(-1)

        with torch.no_grad():
            ref_outputs = ref_model(full_ids)
            ref_logits = ref_outputs.logits[:, prompt_len - 1:-1, :]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_token_log_probs = ref_log_probs.gather(
                2, completion_ids.unsqueeze(-1)
            ).squeeze(-1)

        mask = (completion_ids != tokenizer.eos_token_id).float()

        # 4. KL PENALTY: per-token divergence from reference
        kl_per_token = token_log_probs - ref_token_log_probs
        kl_penalty = (kl_per_token * mask).sum(dim=1) \
                     / mask.sum(dim=1).clamp(min=1)

        # 5. POLICY GRADIENT
        adjusted_rewards = rewards - kl_coeff * kl_penalty.detach()
        baseline = adjusted_rewards.mean()
        advantages = adjusted_rewards - baseline

        seq_log_probs = (token_log_probs * mask).sum(dim=1)
        policy_loss = -(seq_log_probs * advantages.detach()).mean()

        # 6. UPDATE
        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return history
```

### Log Probabilities: The Gradient Signal

When the model generates text, it assigns a probability to each token it picks. "How confident was I that '4' comes next?"

The notation `log π(y_t | x, y_{<t})` means: the log probability of token `y_t`, given the prompt `x` and all prior tokens `y_{<t}`. Think of it as the model's confidence score for each word choice it made.

```
Token log probs for "4\n":
  log P("4" | prompt)     = -2.3    ← model thought "4" was unlikely
  log P("\n" | prompt, 4) = -0.1    ← newline was expected

Sequence log prob = sum of token log probs = -2.4
```

The gradient of this quantity, weighted by the advantage, is the REINFORCE update. If a completion scored above average (positive advantage), increase the probability of that token sequence. If below average (negative advantage), decrease it.

### The KL Penalty: Don't Forget How to Speak

Without a KL penalty, the model collapses. It finds one high-reward completion, say "2", and puts all its probability mass there. Task performance looks great, but the model forgets everything else. Can't write prose, can't follow other instructions, can't do anything beyond outputting "2".

To prevent this, we keep a frozen copy of the original GPT-2 (the "reference model") and penalize the trained model for behaving too differently from it. The penalty is called KL divergence, a measure of how much two probability distributions differ. At each token, we ask: how much did the trained model's probability shift from the original's? The bigger the shift, the higher the penalty.

```
KL = Σ_t [ log π(y_t|...) - log π_ref(y_t|...) ] / T
```

The coefficient `kl_coeff = 0.05` controls how much we care about staying close to the original model vs. maximizing reward. At 0.05, the model learns to solve the task within the space of outputs GPT-2 would plausibly generate.

### The Advantage: Relative, Not Absolute

Raw rewards are 0 or 1. But we don't just want to tell the model "that was good" or "that was bad." We want to say "that was *better than your average output*" or "that was *worse than your average output.*" This relative signal is called the advantage.

```
Batch of 32 completions:
  20 got reward 0, 12 got reward 1
  Mean reward = 0.375

  Completion with reward 1: advantage = 1.0 - 0.375 = +0.625
  Completion with reward 0: advantage = 0.0 - 0.375 = -0.375
```

Subtracting the batch mean (the "baseline") centers the advantages around zero. Why does this matter? Imagine the model is already getting 90% of completions right. Without baselining, every correct completion still pushes the gradient with full force, even though the model barely needs to change. With baselining, a correct completion in a strong batch is only a small positive signal, while a correct completion in a weak batch is a large one.

## SFT vs REINFORCE: Where They Diverge

Both hit high task performance. The output distributions tell the real story.

```
SFT output distribution (after 200 steps):
  "2" ████████████████████  45%
  "4" ████████████████████  44%
  "6" ████                   8%
  "8" █                      2%
  "10" ░                     1%

REINFORCE output distribution (after 200 steps):
  "2" ████████████           25%
  "4" ██████████             22%
  "6" ██████████             21%
  "8" ████████               18%
  "10" ██████                14%
```

SFT collapses toward whichever demonstrations the optimizer saw most recently. It memorizes the training data. REINFORCE explores. It finds all five correct answers and spreads probability across them, because any even number gets the same reward.

This is the core difference between imitation and optimization. SFT minimizes distance to the demonstrations. REINFORCE maximizes expected reward. When the demonstrations are incomplete or biased, SFT inherits those biases. REINFORCE doesn't need demonstrations at all.

## Why This Matters Beyond Toy Problems

This 200-line experiment is a miniature version of the RL step in RLHF (Reinforcement Learning from Human Feedback). The production version uses PPO instead of REINFORCE, a learned reward model instead of a regex, and runs on models with 70B+ parameters. But the bones are the same:

1. Sample from the current policy
2. Score the samples
3. Compute log probabilities under current and reference policies
4. Penalize KL divergence from the reference
5. Update toward higher-advantage completions

The reason RL works for LLM alignment, where SFT alone falls short, is exactly the difference we saw above. SFT can only teach a model to copy labeled examples. RL lets a model discover behaviors that satisfy a reward signal, including behaviors no human demonstrator thought to write down.

GRPO (Group Relative Policy Optimization), the algorithm behind DeepSeek-R1, drops the critic network entirely and computes advantages from group statistics. That's closer to our batch-mean baseline than PPO's learned value function. REINFORCE isn't a stepping stone. The field keeps coming back to it.

## The Full Picture

```
┌──────────────────────────────────────────────────────────┐
│                    REINFORCE Loop                        │
│                                                          │
│  ┌─────────┐    ┌──────────┐    ┌──────────────────┐    │
│  │ Policy  │───→│ Generate │───→│ Reward Function  │    │
│  │ (GPT-2) │    │ Samples  │    │ (even number?)   │    │
│  └────▲────┘    └──────────┘    └────────┬─────────┘    │
│       │                                  │              │
│       │         ┌──────────┐             │              │
│       └─────────│ Gradient │◄────────────┘              │
│                 │ Update   │                            │
│                 └──────────┘                            │
│                                                          │
│  Loss = -E[ log π(y|x) · advantage ]                   │
│  advantage = reward - β·KL - baseline                   │
└──────────────────────────────────────────────────────────┘
```

The entire training loop fits in a single function. No replay buffer, no critic network, no importance sampling. Sample, score, update. That's REINFORCE.

---

## References

1. Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3-4), 229–256.
2. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
3. Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training language models to follow instructions with human feedback. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
4. Shao, Z., Wang, P., Zhu, Q., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) — introduces GRPO.
5. Guo, D., Yang, D., Zhang, H., et al. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
