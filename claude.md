Yes—let’s distill *AgentOccam* into a specification-style summary, where each innovation is treated like a **designable, engineerable feature**. Think of it as a spec doc for an LLM web agent system.

---

# 📄 AgentOccam: Engineering Specification Summary

## **Overview**

**Goal:** Improve LLM-based web agent performance by aligning the agent’s *observation* and *action* spaces with what LLMs are naturally good at (i.e., reading/writing static text).

**Key Principles:**

* Avoid handcrafted agent policies
* No in-context examples, no online search
* Leverage LLM zero-shot capabilities
* Minimize unnecessary embodiment complexity

---

## **Feature 1: Observation Space Restructuring**

### **Feature Name:** `ObservationSimplifier`

#### **Inputs**

* Raw HTML or accessibility tree from browser environment
* Prior interaction history and current task goal

#### **Processing Steps**

* Remove:

  * Redundant `StaticText` next to `link`
  * Visual-only formatting elements (e.g. `gridcell`, `columnheader`)
* Convert structured content (e.g., tables/lists) to Markdown
* Token budget management:

  * Prioritize “pivotal” nodes and their {ancestor, sibling, descendant} context
  * Replay only task-relevant history tied to current plan

#### **Outputs**

* A compact, plain-text rendering of the web page:

  * Focused on semantically meaningful content
  * Easy for LLMs to read and reason over

#### **Rationale**

Improves semantic alignment with LLM pretraining distribution; reduces token bloat and noise.

---

## **Feature 2: Action Space Reduction and Alignment**

### **Feature Name:** `ActionReducer`

#### **Inputs**

* Full set of valid WebArena/WebVoyager browser actions

#### **Processing Steps**

* Remove:

  * Low-utility actions (e.g. `noop`, `tab_focus`, `scroll`, `goto`)
  * Embodiment-heavy actions (e.g. `hover`, `press`)
* Retain only core interaction actions:

  * `click`, `type`, `go_back`, `go_home`
* Add:

  * `note` (record intermediate insights)
  * `stop` (terminate with answer)
  * `branch` / `prune` (see Feature 3)

#### **Outputs**

* A compact, high-signal action set suitable for text-based LLM reasoning

#### **Rationale**

Reduces distraction and missteps by removing rarely-used or ill-suited operations

---

## **Feature 3: Planning Tree Workflow**

### **Feature Name:** `SelfPlanningAgentTree`

#### **Inputs**

* Task intent
* Current goal state
* Action history

#### **Processing Steps**

* Allow agent to `branch` at any point: define subgoals under current plan
* Allow agent to `prune` if current plan is failing: revert to prior decision point
* Discard irrelevant historical steps from previous plans when replaying context

#### **Outputs**

* Tree-structured plan with metadata at each node:

  * Goal
  * Rationale
  * Action trace

#### **Rationale**

Mimics hierarchical planning without requiring explicit memory modules or managers; improves consistency and task recovery.

---

## **Feature 4: Prompt Design for Stateless Inference**

### **Feature Name:** `FlatPromptPackager`

#### **Prompt Layout**

```
General Instruction:
"You are a web agent. Issue a correct action for the current step."

Task Description:
"Find all reviews that mention 'battery life'."

Online Task Info:
- Current Subgoal: "Check product reviews"
- Previous Actions: (summarized trace)
- Current Observation: (cleaned DOM via ObservationSimplifier)

Action Specification:
- click [id]
- type [id] [text]
- branch [intent]
- prune [reason]
- ...
```

#### **Design Goals**

* Use fixed template across all tasks and sites
* Avoid need for few-shot or multi-role prompting
* Stay within context window budget

---

## **Deployment Notes**

* **LLM API**: GPT-4-turbo (zero-shot, no tuning)
* **Evaluation**: WebArena (812 tasks, 5 domains)
* **Performance Gains**:

  * +9.8 pts over previous SOTA (29% rel.)
  * +26.6 pts over vanilla web agent (161% rel.)

---

Want me to expand this into a full doc template (e.g., with metrics, UX hooks, or implementation notes)? Or tailor it for a particular team context (e.g., infra, agent dev, eval)?
