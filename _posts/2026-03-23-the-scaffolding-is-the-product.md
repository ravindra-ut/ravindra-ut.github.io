---
layout: post
title: "The Scaffolding Is the Product: Why Browser Agent Harnesses Matter More Than the Model"
date: 2026-03-23
categories: [AI, Agents, Systems]
tags: [llm, browser-agents, harness-engineering]
---

*Why the engineering around the model often matters more than the model itself.*

---

## I. The Obvious Thing That Doesn’t Work

Watch an LLM try to use a website for the first time. It clicks a loading spinner. It types into a disabled field. It retries both, then confidently announces success.

These failures have a pattern—and they deserve names:

- **Phantom clicking**: targeting elements that don’t exist or aren’t interactable  
- **State blindness**: not knowing what changed after an action  
- **Action loops**: retrying the same failed interaction with no awareness of failure  

These are **primarily failures of perception**, not reasoning. The model isn’t confused about *what* to do—it’s confused about *what it sees*.

A typical webpage contains thousands of DOM nodes. Only a tiny fraction are actually usable. Handing the full DOM to a language model is like handing someone a phone book and asking them to order dinner.

The intelligence is not the bottleneck. The interface is.

---

## II. The Shift: From Language to Structured Action

Earlier attempts at browser agents relied on natural language:

> “Click the blue button on the left”

This interface is fundamentally ambiguous.

Modern systems instead use structured tool calls:

```
click_element_by_index(37)
```

This changes everything.

The model’s outputs become:
- **constrained**
- **interpretable**
- **guaranteed to execute**

This shift—from language to structured action—is what makes reliable agent systems possible.

---

## III. The Core Idea: The Harness Is the Product

Systems like PageAgent don’t primarily improve the model.  
They improve:

- what the model **sees**
- how the model **acts**
- what the model is **forced to think about**

This surrounding system—the **harness**—is not just glue code.

It is where the system’s *operational intelligence* lives.

---

## IV. Four Layers of Compression

The harness transforms a chaotic webpage into something a model can reason about.

Think of it as a pipeline:

```
Raw DOM → Filtered → Indexed → Diffed → Act
```

Each step removes noise and adds structure.

---

### 1. Filtering (solving phantom clicking)

The first step reduces the DOM to its **interactive skeleton**.

An element survives only if it is:
- visible  
- within or near the viewport  
- interactive (inputs, buttons, ARIA roles, event handlers, cursor hints)

Everything else is discarded.

Then attributes are stripped down to only what matters:
- `placeholder`
- `aria-label`
- `role`
- `value`
- state indicators (`checked`, `expanded`)

A 5,000-node DOM becomes ~30 usable elements.

This compression is aggressive—but *lossy in the right places*.

---

### 2. Indexing (giving the model hands)

Each remaining element is assigned an index:

```
[35] Email input
[36] Password input
[37] "Sign In" button
```

Instead of describing elements, the model acts directly:

```
click_element_by_index(37)
```

This does two things:
- eliminates ambiguity
- makes invalid actions impossible

The harness defines what is clickable.  
The model decides what to click.

---

### 3. Diffing (solving state blindness)

After every action, the system marks **what changed**.

New elements are explicitly annotated:

```
[40] "Next" button
*[45] Search results
*[46] Result: Flight to London
```

The `*` means: *this did not exist before your last action.*

This is the critical shift:

> The model no longer has to infer change—it is told what changed.

No fragile mental diffing. No guesswork.

---

### 4. Reflection (solving action loops)

Before every action, the model is forced to think:

- **Did my last action work?**
- **What should I remember?**
- **What is my next goal?**

This is enforced structurally—not suggested.

Example:

```
evaluation_previous_goal: Form submission failed due to invalid email
memory: Email field requires valid format
next_goal: Correct email and resubmit
```

This prevents blind retries.

It turns the agent from reactive to **self-correcting**.

---

## V. A Concrete Example

Consider a simple login flow:

1. Model sees:
   - Email field
   - Password field
   - Sign In button

2. It enters an invalid email and clicks Sign In

3. The next state includes:
   ```
   *[42] "Invalid email address" error
   ```

4. The reflection step forces:
   - recognition of failure  
   - memory of the constraint  
   - correction of the plan  

Without diffing + reflection, the model often retries blindly.

With them, it adapts immediately.

---

## VI. One Action Per Step

Most systems allow multiple actions per turn.

This system allows exactly one.

Why?

Because the web is a **reactive environment**.

Clicking a button might:
- navigate
- open a modal
- trigger async loading
- do nothing

You cannot safely plan multiple steps ahead.

Batching actions turns a closed-loop system into an **open-loop controller** in a stochastic environment.

So the agent must:

> act → observe → think → act

---

## VII. The Harness as Co-Pilot

The system also injects lightweight guidance:

- Detects navigation → “Page changed to …”
- Prevents excessive waiting
- Adds urgency near step limits

These are not intelligent behaviors.

But they shape the agent’s trajectory in ways that look like intelligence.

---

## VIII. What This Doesn’t Solve

The approach has real limitations:

- **Lossy filtering** can hide valid interactions  
- **Custom JS-heavy components** may break detection  
- **One-step execution is slow**  
- **Reflection adds overhead**  
- **Step limits constrain complex tasks**  

The harness embeds assumptions about what matters—and those assumptions can be wrong.

---

## IX. The General Principle

This pattern extends beyond browser agents:

> When deploying models in complex environments, the harness matters as much as the model.

Empirically:
- Systems with identical models perform very differently depending on harness design

Conceptually:
- This aligns with the extended mind thesis—tools become part of cognition

In this view:
- Diffing = perception  
- Memory = working memory  
- Reflection = reasoning structure  

The model supplies capability.  
The harness makes that capability usable.

---

## X. Closing Thought

We often ask:

> “How do we make models smarter?”

But for agents, a better question is:

> “What kind of world are we giving the model to think in?”

If you want better agents, don’t just build better models.

**Build better worlds for them to operate in.**

---

## References

1. Zhou et al., WebArena, ICLR 2024  
2. Drouin et al., WorkArena, 2024  
3. Wei et al., Chain-of-Thought Prompting, NeurIPS 2022  
4. Clark & Chalmers, The Extended Mind, 1998  

---