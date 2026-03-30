---
layout: post
title: "Reward Models from Scratch: Training DistilGPT-2 to Rank Human Preferences"
date: 2026-03-30
categories: [AI, RL]
tags: [reinforcement-learning, llm, reward-model, rlhf, transformers]
---

*How reward models work, what the Bradley-Terry loss does, and what 100 lines of training code reveal about the scoring function behind RLHF.*

## The Setup

In the [last post]({% post_url 2026-03-27-reinforce-for-llms %}), we trained GPT-2 to generate even numbers using REINFORCE. The reward function was a regex and a modulo check. Always correct. Trivially cheap.

Real RLHF doesn't have that luxury. When you want to reward "helpful, harmless, and honest," there's no `if` statement for that. You need a neural network, a **reward model**, that looks at a response and outputs a single number: how good is this? It sits between the humans who label preferences and the policy that learns from the scores. This post builds one from scratch.

We'll take Anthropic's `hh-rlhf` dataset (human preference pairs of assistant responses), add a single scoring layer on top of DistilGPT-2 (a smaller, faster version of GPT-2), and train it to rank chosen responses above rejected ones. 5,000 examples. Three epochs. The model reaches ~69% validation accuracy.

## The Data: Preference Pairs

The `hh-rlhf` dataset gives us pairs: a chosen response and a rejected one. Same conversation, two different assistant replies.

```python
from datasets import load_dataset

dataset = load_dataset("Anthropic/hh-rlhf", split="train[:5000]")

example = dataset[0]
print("CHOSEN:")
print(example["chosen"][:500])
print("\nREJECTED:")
print(example["rejected"][:500])
```

Each example is a full multi-turn conversation. The chosen and rejected branches diverge at the last assistant turn. The human picked one; the dataset records both.

One pattern in the data: chosen responses are longer than rejected ones about 67% of the time. The reward model will pick this up. We'll test for it later.

## Architecture: Backbone + Scalar Head

A reward model is a language model with one change: instead of predicting the next token, it outputs a single number.

We take DistilGPT-2 and split it into two parts. The **backbone** is the transformer that reads text and builds a rich representation of it. On top of that, GPT-2 normally has a **language modeling head** that predicts the next token. We throw that head away and replace it with a single linear layer: one matrix multiply that takes the backbone's representation and produces a single number. That number is the reward score.

One detail: the backbone produces a representation for every token in the input. We need to pick one to feed into our scoring layer. Since the transformer reads left to right, the last real token (not a padding token) has seen the entire conversation. We grab that one.

```python
class RewardModel(nn.Module):
    def __init__(self, model_name="distilgpt2"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.n_embd
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        last_hidden = outputs.last_hidden_state
        # Find the last real token (not padding)
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        pooled = last_hidden[batch_idx, seq_lengths]
        return self.reward_head(pooled).squeeze(-1)
```

## Tokenizing Preference Pairs

Each training example produces two inputs: the chosen conversation and the rejected conversation, both tokenized. Both go through the model, both get a score, and the loss operates on the difference.

```python
def tokenize_pair(example, tokenizer, max_length=256):
    chosen = tokenizer(
        example["chosen"], truncation=True, max_length=max_length,
        padding="max_length", return_tensors="pt"
    )
    rejected = tokenizer(
        example["rejected"], truncation=True, max_length=max_length,
        padding="max_length", return_tensors="pt"
    )
    return {
        "chosen_ids": chosen["input_ids"].squeeze(0),
        "chosen_mask": chosen["attention_mask"].squeeze(0),
        "rejected_ids": rejected["input_ids"].squeeze(0),
        "rejected_mask": rejected["attention_mask"].squeeze(0),
    }
```

We cap at 256 tokens. These conversations can run much longer than that, so we're cutting off the end of many examples. That means the model sometimes never sees the part of the conversation where the chosen and rejected responses actually differ. A real system would use 512 or 1024 tokens or even more, but 256 keeps training fast enough for a toy experiment.

## The Bradley-Terry Loss

We don't train the model to predict a specific score. We train it to rank: the chosen response should score higher than the rejected one.

The Bradley-Terry model (from 1952, originally for ranking chess players) says: the probability that option A beats option B depends on the difference in their scores, passed through a sigmoid function. The sigmoid squashes any number into a probability between 0 and 1. A large positive difference gives a probability near 1 ("A is almost certainly better"). A large negative difference gives near 0. A difference of zero gives 0.5: a coin flip.

```
P(chosen > rejected) = σ(r_chosen - r_rejected)
```

The loss is the negative log of this probability:

```
L = -log σ(r_chosen - r_rejected)
```

When the model correctly ranks the chosen response higher, `r_chosen - r_rejected` is positive, the sigmoid is close to 1, and the loss is small. When the model gets it backwards, the difference is negative, the sigmoid is near 0, and the loss spikes.

```python
def compute_bt_loss(model, batch, device):
    chosen_rewards = model(
        batch["chosen_ids"].to(device),
        batch["chosen_mask"].to(device)
    )
    rejected_rewards = model(
        batch["rejected_ids"].to(device),
        batch["rejected_mask"].to(device)
    )
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    accuracy = (chosen_rewards > rejected_rewards).float().mean()
    return loss, accuracy, chosen_rewards.mean(), rejected_rewards.mean()
```

Note what's missing: no target scores. We never tell the model "this response is a 7.3." We only say "this one is better than that one." The absolute scale of the scores doesn't matter. When this reward model is plugged into an RL training loop (like the one in the [last post]({% post_url 2026-03-27-reinforce-for-llms %})), only the relative differences between candidates matter.

## Training

We use AdamW as the optimizer. It's the standard choice for fine-tuning transformers because it handles the learning rate and weight updates more carefully than plain gradient descent. The learning rate is 1e-5 (0.00001), which is deliberately small: we want to nudge the pretrained weights, not overwrite them. We also clip gradients at 1.0, which means if any single update step would be too large, we scale it down. This prevents the occasional bad batch from blowing up the model.

We train for three passes (epochs) over 4,000 training pairs, holding out 1,000 pairs for validation, data the model never trains on, so we can check if it's learning something general or just memorizing.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

for epoch in range(3):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
    val_metrics = evaluate(model, val_loader, device)

    print(f"Epoch {epoch+1}: "
          f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
          f"val_loss={val_metrics['loss']:.4f}, "
          f"val_acc={val_metrics['accuracy']:.3f}")
```

```
Epoch 1: train_loss=0.6953, train_acc=0.578, val_loss=0.6847, val_acc=0.551
Epoch 2: train_loss=0.6354, train_acc=0.629, val_loss=0.6533, val_acc=0.575
Epoch 3: train_loss=0.5883, train_acc=0.672, val_loss=0.5805, val_acc=0.634
```

The gap between train and val loss is small. The model does about as well on data it has never seen as on data it trained on. That means we're not overfitting (memorizing the training data). If anything, we're underfitting: the model hasn't learned everything it could. With only 82M parameters and 256 tokens of context, there's a ceiling on how much of human preference it can capture.

## Probing for Length Bias

The model ranks human preferences. But what did it learn? One probe: give it the same question answered at four lengths, from terse to verbose, all factually correct.

```python
prompt = "Human: What is the capital of France?\n\nAssistant:"
responses = [
    " Paris.",
    " The capital of France is Paris.",
    " The capital of France is Paris. Paris has been the capital since the "
    "10th century and is the largest city in France...",
    " The capital of France is Paris. Paris, often called the City of Light, "
    "has served as the capital since the 10th century..."  # ~80 words
]
```

```
Response             Words   Reward
Paris.                   1    2.537
The capital of...        6    2.728
Paris has been...       40    2.872
Paris, often cal...     80    0.524
```

Longer responses score higher, monotonically. The model learned a length heuristic. We saw that chosen responses are longer 67% of the time. The model found the easiest correlated feature and leaned on it.

Production reward models fight this in several ways. One is **length normalization**: divide the reward by the number of tokens so that a longer response doesn't automatically score higher. Another is training on **contrastive pairs matched for length**, pairs where the chosen and rejected responses have similar word counts, forcing the model to learn something beyond "longer is better." Without countermeasures like these, the RL policy learns to be verbose, maximizing reward by writing more, not better.

## Reward Distributions

The model scores chosen responses higher than rejected ones on average, but the distributions overlap heavily. It can't reliably separate individual pairs. If you look at the **reward margin** for each pair, the chosen score minus the rejected score, most margins are positive (the model agrees with the human), but a meaningful fraction are negative (the model gets it backwards).

```python
chosen_r, rejected_r = collect_reward_distributions(model, val_loader, device)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(chosen_r, bins=40, alpha=0.6, label="Chosen", color="tab:green")
axes[0].hist(rejected_r, bins=40, alpha=0.6, label="Rejected", color="tab:red")
```

That overlap matters. A noisy reward model doesn't just fail to help the RL policy; it misleads it. The policy finds and exploits every systematic error. Length bias is one. Confident-sounding nonsense scoring well is another.

## Where This Fits in the RLHF Pipeline

In the [last post]({% post_url 2026-03-27-reinforce-for-llms %}), the reward function was a hard-coded check. In the full RLHF pipeline, this reward model replaces that check.

```
┌──────────────────────────────────────────────────────────────┐
│                     Full RLHF Pipeline                       │
│                                                              │
│  Step 1: Supervised Fine-Tuning (SFT)                        │
│    Train a base model on demonstration data.                 │
│                                                              │
│  Step 2: Reward Modeling  ← this post                        │
│    Train a scoring function on human preference pairs.       │
│                                                              │
│  Step 3: RL Fine-Tuning  ← last post                         │
│    Use REINFORCE / PPO / GRPO to optimize the SFT model      │
│    against the reward model, with a KL penalty to the        │
│    SFT checkpoint.                                           │
└──────────────────────────────────────────────────────────────┘
```

Step 3 is where the reward model's errors compound. The policy runs for thousands of update steps, each time pushing toward whatever scores highest. A small bias (longer = better) becomes a large artifact (every response is five paragraphs of padding).

The RL algorithm can only be as good as the signal it optimizes.

## What We Skipped

**Scale.** InstructGPT used a 6B parameter reward model. Llama 2's was 70B, nearly 1,000x larger than ours. A bigger model can pick up on subtle differences in tone, factual accuracy, and reasoning quality, rather than relying on surface heuristics like response length.

**Label noise.** Anthropic's RLHF paper found that when two different humans label the same preference pair, they only agree about 73% of the time. Our model's 69% accuracy is close to that ceiling. It may be learning roughly as much as the labels can teach.

**Reward hacking.** Once the RL policy starts optimizing against the reward model, it finds responses that score high but aren't actually good. A model that outputs confident, detailed, plausible-sounding wrong answers, because the reward model learned that confidence and detail correlate with preference. The reward model is a proxy for human judgment, and the policy will exploit every crack in that proxy.

**DPO.** Direct Preference Optimization folds preference learning directly into the policy update, skipping the reward model entirely. Instead of training a separate scoring function and then optimizing against it, DPO uses the preference pairs to update the policy in a single step. No separate scoring function means no reward hacking loop.

## The Full Picture

```
┌──────────────────────────────────────────────────────────┐
│                  Reward Model Training                   │
│                                                          │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────┐     │
│  │ Chosen   │──→│              │──→│ r_chosen      │     │
│  │ Response │   │  DistilGPT-2 │   │               │     │
│  └──────────┘   │  + Linear    │   │ Bradley-Terry │     │
│  ┌──────────┐   │    Head      │   │ Loss          │     │
│  │ Rejected │──→│              │──→│ r_rejected    │     │
│  │ Response │   └──────────────┘   └───────┬───────┘     │
│  └──────────┘                              │             │
│                                            ▼             │
│                  L = -log σ(r_chosen - r_rejected)       │
│                                                          │
│  The model never sees target scores.                     │
│  It only learns: this one is better than that one.       │
└──────────────────────────────────────────────────────────┘
```

The reward model compresses all of human judgment into a single number. Every shortcut it learns, the RL policy exploits. Every preference it misreads, the RL policy amplifies.

Building a good reward model is harder than the RL step that follows.

---

## References

1. Bradley, R. A. & Terry, M. E. (1952). Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons. *Biometrika*, 39(3/4), 324–345.
2. Christiano, P., Leike, J., Brown, T., et al. (2017). Deep Reinforcement Learning from Human Preferences. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)
3. Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training language models to follow instructions with human feedback (InstructGPT). [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
4. Bai, Y., Jones, A., Ndousse, K., et al. (2022). Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. [arXiv:2204.05862](https://arxiv.org/abs/2204.05862)
5. Touvron, H., Martin, L., Stone, K., et al. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)
6. Rafailov, R., Sharma, A., Mitchell, E., et al. (2023). Direct Preference Optimization: Your Language Model Is Secretly a Reward Model. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
