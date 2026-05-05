# COCA Prompts

## Critic Model for Critique Generation

```text
Please evaluate the model’s answer to the user’s input and generate critique for improvement. When generating the critique, please note:
  - The critique should be concise and clear, easy for the model to understand.
  - If the model’s answer is correct, the critique should simply be: "No corrections needed."

Now, review the user’s input and model’s answer:
User Input:
<User Input>

Model’s Answer:
<Model Generated Content>
```

## Actor Model for Reasoning Refinement

```text
Please refine the model-generated response by considering the corrective feedback provided below. These feedback serve as guidance to enhance the accuracy, relevance, and comprehensiveness of the response. While incorporating the suggestions, ensure that the final output remains natural and contextually appropriate, rather than strictly adhering to the feedback.

User Input:
<User Input>

Model Generated Content:
<Model Generated Content>

Corrective Feedback:
<Corrective Feedback>

Based on the corrective feedback, refine the model's response while maintaining natural flow and coherence. Use the feedback as helpful suggestions rather than strict rules. The final response should be improved but still retain its original style and intent. Output the revised content directly (Note: the output should not include any prefixes or suffixes like <–Start> or <–End>).
```
