---
layout: post
title: "The Scaffolding Is the Product: How Browser Agent Harnesses Do the Real Work"
date: 2026-03-23
categories: [AI, Agents, Systems]
tags: [llm, browser-agents, harness-engineering]
---

*Why the engineering around the model matters more than the model itself.*

## The Obvious Thing That Doesn't Work

Watch an LLM try to use a website for the first time. It clicks a loading spinner. It types into a disabled field. It retries both, three times each, then announces it has completed the task.

These failures have a pattern, and they deserve names. *Phantom clicking*: targeting elements that don't exist or can't be interacted with. *State blindness*: not knowing what changed after an action, so the model can't tell whether a dropdown opened or a form submitted. *Action loops*: retrying the same failed interaction because nothing in the model's input signals that it already tried and failed. These are not failures of reasoning. They are failures of *perception*, and no amount of model scaling fixes them. The problem is not in the model. It is in what the model sees.

A typical webpage contains 5,000 DOM nodes: layout containers, invisible styling elements, tracking pixels, decorative wrappers. Fewer than 40 of them are things a user would ever interact with. Handing this raw tree to a language model is like handing a phone book to someone and asking them to order dinner. The intelligence is not the bottleneck. The interface is.

[PageAgent](https://github.com/alibaba/page-agent), a browser automation agent by Alibaba, takes a different approach. Rather than improving the model, it improves what the model sees, how it acts, and what it's forced to think about between actions. The entire architecture is a *harness*, a structured scaffold that compresses a webpage into something a language model can reason about, then translates the model's decisions back into browser interactions. I want to argue that this harness is not merely an implementation detail. It is, in a meaningful sense, *where the intelligence lives*.

## Why This Became Possible

Browser agents have been attempted for years, but harness engineering of this kind only became viable recently. Early web automation with language models relied on free-form text generation: the model would produce something like "click the blue button on the left," and a brittle parser would try to resolve that to a DOM element. The failure rate was high because the interface between model and browser was *linguistic*, full of ambiguity.

Modern LLMs support structured output: tool-call schemas, typed parameters, constrained generation. When a model can emit `click_element_by_index(37)` as a structured tool call rather than a natural language instruction, the harness can guarantee that every output maps to a valid browser action. The model's execution loop becomes observable, interceptable, and constrainable. PageAgent is built on top of this shift.

## Four Layers of Compression

The harness transforms the DOM through four progressive layers, each narrowing what the model sees while preserving what it needs:

```
Raw DOM (everything) → Filtered (what matters) → Indexed (what you can do) → Diffed (what just changed)
```

Each layer solves a named failure mode.

### Filtering: solving phantom clicking

PageAgent's first pass strips the DOM to its interactive skeleton. An element survives only if it is visible, within or near the current viewport, and interactive, which includes recognized input types, elements with interactive ARIA roles, click-related event handlers, or cursor styles that signal clickability. Everything else is discarded.

The surviving elements are simplified further. A raw HTML button might carry thirty attributes: CSS classes, test identifiers, analytics hooks, inline styles. The harness keeps only attributes that carry semantic meaning for a reasoning agent: `placeholder`, `aria-label`, `type`, `value`, `role`, and state indicators like `checked` and `aria-expanded`. Attributes that repeat the element's visible text or duplicate a parent's attribute are removed. A 5,000-node DOM becomes roughly 30 elements. The compression ratio often exceeds 100:1, *lossy in the right places*, discarding implementation noise while preserving interaction semantics.

Prior work on web agents, notably WebArena [1] and browser-gym environments [2], performs similar DOM preprocessing. PageAgent's contribution is in the specificity of its filtering heuristics. The combination of cursor-style detection, event listener introspection, ARIA role matching, and visibility checks is tuned for real-world (not benchmark) webpages, where the line between interactive and decorative is often blurry.

### Indexing: giving the model hands

Compression solves perception, but the model still needs a way to act. A human points and clicks. A language model can only produce text. PageAgent bridges this by assigning every surviving element a numeric index:

```
[33] User form
  [35] Email input (placeholder: "Email")
  [36] Password input
  [37] "Sign In" button
```

To click the sign-in button, the model emits `click_element_by_index(37)`. The harness resolves the index to the DOM node and dispatches the event. No CSS selectors, XPaths, or coordinate pairs. Because only real, visible, interactive elements receive indices, the model is *structurally prevented* from phantom clicking. The harness decides what is clickable; the model decides what to click.

PageAgent also draws colored numbered overlays on the actual webpage, making the model's perception transparent to the human observer. When the model clicks element 37, the user can look at the screen and see exactly which element that is. This transforms the agent from a black box into something visually auditable in real time.

### Diffing: solving state blindness

This is the most interesting layer. After every action, the model needs to answer a fundamental question: *did what I just did work?* It receives a fresh list of elements at each step. Without help, it must mentally diff this list against its memory of the previous one, a task language models perform unreliably as lists grow longer.

PageAgent closes this loop with an explicit diff signal. The harness maintains a `WeakMap` keyed by DOM node references. Each time the element tree is rebuilt, every interactive element is checked against this cache. Previously seen elements appear normally. Elements appearing for the first time are marked with an asterisk:

```
[40] "Next" button
*[45] Search results
*[46] Result: Flight to London
*[47] Result: Flight to Paris
```

That asterisk tells the model: these elements did not exist before your last action. The model knows instantly that its search produced results. No mental diffing required.

The `WeakMap` choice is quietly elegant. When elements leave the DOM, the runtime garbage-collects the references and the cache entries disappear with them. If those elements reappear later as new instances (which is how frameworks like React operate), they are correctly identified as new. The cache is self-cleaning and framework-aware without explicit framework integration. Most browser agent architectures rely on the model's own context window to track changes between steps. PageAgent's explicit diff annotation offloads this cognitive burden to the harness, eliminating state blindness as a failure class at almost no computational cost.

### Reflection: solving action loops

The final layer is not perceptual but cognitive. At every step, PageAgent forces the model to produce a structured reflection *before* acting, consisting of three fields: an **evaluation** of whether the last action succeeded or failed, a **memory** scratchpad for information to carry forward (confirmation numbers, partial results, constraints discovered), and a **next_goal** stating what the model intends to do and why. The schema enforces this. The model cannot bypass it.

This directly prevents action loops. A model that must evaluate its previous action before choosing the next one is forced to confront failure rather than blindly retry. The design draws on a well-established finding: models reason more accurately when they externalize their chain of thought before committing to a decision [3]. PageAgent goes further by structuring the reflection into functionally distinct components, so that backward-looking assessment, information management, and forward-looking planning happen as separate, enforced steps.

The memory field deserves particular attention. In multi-step web tasks, information encountered early often becomes critical later. A confirmation code. A price seen on page one that must be compared on page three. The memory field provides a curated, self-maintained summary, applying *selection pressure*: the model must decide what to remember, which means it must decide what matters.

## How It Plays Out: A Login Flow

Consider what happens when the agent encounters a login form. It sees three indexed elements: an email field, a password field, a sign-in button. It enters an invalid email and clicks submit.

The next state includes a new element: `*[42] "Invalid email address" error`. The asterisk tells the model something appeared. The forced reflection step requires the model to evaluate: "Form submission failed, a validation error appeared on the email field." Its memory records the constraint. Its next goal becomes: "Correct the email format and resubmit."

Without diffing, the model would see the same form and might retry the same input. Without reflection, it would have no structured moment to recognize the failure. With both, it adapts on the first try. The harness didn't make the model smarter. It gave the model the information and the cognitive structure needed to use the intelligence it already has.

## One Action Per Step

Most LLM tool-use frameworks allow the model to call multiple tools per turn. PageAgent takes the opposite approach: all available actions are bundled into a single macro-tool, and the model must select exactly one. Click, type, select, scroll, wait, execute JavaScript, ask the user, or declare done. One decision, then observe the result.

This constraint exists because the web is reactive in ways that are difficult to predict. Clicking a button might submit a form, trigger a navigation, open a modal, or do nothing visible for several seconds while an API call completes. The correct next action depends entirely on which outcome occurred, and the model cannot know without observing first. Batching actions would amount to open-loop control in a closed-loop environment. Every action gets a full observe-reflect-act cycle.

## The Harness as Co-Pilot

Beyond perception and action, the harness monitors the agent's behavior and intervenes when it detects patterns associated with known failure modes. Three heuristics are currently implemented. When the page URL changes, whether the model intended a navigation or not, the harness injects a system observation preventing the model from interacting with stale element indices. When cumulative wait time exceeds a threshold, it warns the model to stop waiting, addressing the tendency to insert wait actions when uncertain. As the step limit approaches, it injects urgency signals at five remaining steps and again at two.

These are simple pattern-matching rules, not intelligence. But they address failure modes that prompt engineering cannot reliably prevent, because the failures arise from the *dynamics* of interaction rather than from the model's understanding of the task.

## What This Doesn't Solve

The DOM simplification is lossy, and sometimes loses the wrong things. Elements whose interactivity is implemented entirely through JavaScript (no ARIA roles, no cursor changes, no standard event handlers) may be filtered out. Custom web components with shadow DOMs present similar challenges. The harness embeds assumptions about what matters, and those assumptions can be wrong.

The one-action-per-step discipline is slow. A human fills a login form in one fluid sequence. The agent observes, reflects, and re-scans between each keystroke. The forced reflection compounds this: the model evaluates, memorizes, and plans before clicking an obvious "Next" button. A cognitive tax a human would not pay. The 40-step default limit is a hard boundary, and complex tasks that legitimately need more steps hit this ceiling and fail. Whether the reliability gains from structured reflection justify the overhead on simple actions is an empirical question that depends on the task distribution.

## The General Principle

The details of PageAgent's harness are specific to browser automation, but the underlying design principle is not. **When deploying a language model as an agent in a complex environment, the engineering of the harness matters at least as much as the choice of model.**

This has empirical support beyond PageAgent. On benchmarks like WebArena, systems using the same base model but different harness designs show dramatically different success rates [1]. A well-designed harness can make a smaller model outperform a larger one operating with less structure. In cognitive science, the *extended mind thesis* [4] argues that cognition does not stop at the skull, that external tools are, functionally, part of the cognitive system. PageAgent's harness is an extended mind for the language model: the diff tracking is an external perceptual system, the memory field is external working memory, the injected observations are an external attention mechanism.

Every browser agent paper benchmarks the model. Almost none benchmark the harness. Yet the harness is doing most of the work. You don't always need a better model. Sometimes you need a better world for the model to operate in.

---

**References**

[1] Zhou, S., Xu, F.F., Zhu, H., et al. "WebArena: A Realistic Web Environment for Building Autonomous Agents." *ICLR 2024.*

[2] Drouin, A., Gasse, M., Caccia, M., et al. "WorkArena: How Capable are Web Agents at Solving Common Knowledge Work Tasks." *2024.*

[3] Wei, J., Wang, X., Schuurmans, D., et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *NeurIPS 2022.*

[4] Clark, A., Chalmers, D. "The Extended Mind." *Analysis, 58(1), 1998.*
