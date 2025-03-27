# Random Popup Idea 💡

```
# There are actually two ways to handle multiple function calls:

# 1. Sequential (One at a time)
Assistant: *makes search call 1*
System: *returns result 1*
Assistant: *analyzes result 1, makes search call 2 if needed*
System: *returns result 2*

# 2. Parallel (Using tool_calls array) 💡 -> how about training with this? each assistant response can have multiple function calls with different search queries
Assistant: *makes multiple search calls at once*
System: *returns all results together*
```
