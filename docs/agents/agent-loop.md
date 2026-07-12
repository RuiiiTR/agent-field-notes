# The agent loop

!!! info "Draft status"
    This article will be completed after the implementation and evaluation traces are available.

An LLM becomes part of an agent when an application repeatedly lets it observe state, select an action, execute a tool, receive the result, and decide whether to continue.

```text
request → context → model → action → tool → observation
                    ↑                         ↓
                    └──────── continue ──────┘
```

## Questions to test

- What terminates the loop?
- Who validates tool arguments?
- What happens when the same action repeats?
- How are tool failures represented?
- Which parts are model behavior and which are application behavior?
