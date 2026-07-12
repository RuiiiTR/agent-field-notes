# Context budgets

A context builder must decide how much space to allocate to instructions, recent messages, selected memories, retrieved evidence, and tool definitions.

```text
instructions + task + history + memory + evidence + tools ≤ context budget
```

The implementation will log the token cost of each component.
