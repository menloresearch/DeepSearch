# Adaptive Search Behavior

- [Agent Action](agent-action.md) -> mostly recognize missing something -> perform "refined query"
- [ ] As a model trainer, I want to inspect the full chat state of the agent to know what's going on so I can improve it -> implement a simple cli inspect tool after training, just print out full chat state.
- Example from AutoDidact:

```markdown
Example Question
What was the reason for substituting the backup Command Module Pilot 3 days prior to the Apollo 13 flight?

Step-by-Step Search Process
Query : "Apollo 13 Command Module Pilot substitution"

Outcome: Retrieved operational support details, but no explanation for the substitution.
Agent's Action: Recognized missing information → **Refined query**.
Query : "Apollo 13 Command Module Pilot substitution reason"

Outcome: Retrieved general mission anomaly details, but still no direct answer.
Agent's Action: **Increased query specificity**.
Query : "Apollo 13 John 'Jack' Swigert substitution"

Outcome: Found general mission reports, but still lacked a clear reason for substitution.
Agent's Action: Hypothesized illness might be a factor → **Refined query** accordingly.
Query : "Apollo 13 Jack Swigert illness substitution"

Outcome: Retrieved the exact explanation: "Several days prior to launch, the backup Lunar Module Pilot became sick with measles. Examinations of the prime crew indicated that the Command Module Pilot was not immune to the disease; therefore, the backup Command Module Pilot was substituted."
Final Answer
The original Command Module Pilot lacked immunity to measles, necessitating his replacement by Jack Swigert.

This example shows how llama learns to do multiple searches to find answers to its questions.
```
