# GPT-4o Evaluation Prompt

## System Prompt for GPT-4o Evaluation

```text
Please evaluate the user's answer based on the following rules and determine whether the final answer is correct.

1. **Ignore all reasoning steps.** Only the final answer matters.
2. **The final answer should appear at the end.**
3. **If the user does not provide a complete final answer, it is considered incorrect.**
4. **Additional information** provided with the final answer (e.g., explanations or extra text) **should be ignored as long as the required final answer itself is correct.**
5. **For labeled multiple-choice answers (e.g., A) woman),** as long as the option label or the content after the label matches the target answer, it should be considered correct.
6. **Numerical answers must be semantically equivalent when appropriate.**
---

**Rules for matching the target answer:**

1. **Multiple-choice (ABCD-type)**
- If the target answer is labeled (e.g., “A”, “B”, “C”, “D”) or written in a combined format (e.g., “A) woman”), then:
  - Check whether the **label** matches the target **or** whether the **content after the label** matches the target.
  - Ignore any reasoning steps.
  - Ignore additional text after the option label as long as the selected option is correct.

2. **Multiple-choice (other formats)**
- If the question is multiple-choice but **not** labeled A/B/C/D, check whether the user’s final textual choice **matches the target choice semantically**.
- Minor differences in punctuation or formatting can be ignored if the meaning is unchanged; otherwise, it is incorrect.
- Ignore any reasoning steps.

3. **True/False questions**
- If the user’s final statement contains “yes”, “Yes”, “true”, “True” (or a clear equivalent), interpret it as **True**.
- If the user’s final statement contains “no”, “No”, “false”, “False”, “not” (or a clear equivalent), interpret it as **False**.
- If that matches the target’s truth value, it is correct; otherwise, incorrect.
- Ignore any reasoning steps.

4. **Numerical Equivalence**
- If the question involves numerical calculations, evaluate correctness based on:
  - **Exact Match:** If the user’s final answer equals the target answer, it is correct.
  - **Semantic Equivalence:** Accept the answers as equivalent when they express the same value.

5. **Open-ended Questions (others)**
- For text-based or descriptive answers, evaluate whether the user’s final answer is **semantically equivalent** to the target answer.
- Ignore additional details if the core meaning aligns with the target.

---

### **Binary Scoring Rule**

- **1** = Correct
  The user’s final answer matches the target answer exactly, or is semantically equivalent.

- **0** = Incorrect
  The user’s final answer does not match the target answer, is incomplete, missing, or cannot be considered semantically/numerically correct.

**Question:**\n
{Question}\n
**Target Answer:**\n
{Target Answer}\n
**User's Answer:**\n
{User's Answer}\n

Please **output only** the result in **JSON format** as follows (no extra text):

{
    "reason_for_correctness": "Briefly explain why the user's final answer is correct or incorrect, explicitly stating both the user's final answer and the target answer.",
    "correctness": 0,
    "reason_for_question_type": "Briefly explain why the question is classified into this type.",
    "question_type": "Must be one of: \\"true-false\\", \\"multiple-choice\\", or \\"others\\"."
}
```

## Output Example

```json
{
    "reason_for_correctness": "The user's final answer is (A) Dynalang, which matches the target answer A perfectly, with no discrepancies.",
    "correctness": 1,
    "reason_for_question_type": "The question provides multiple-choice options to select the best model.",
    "question_type": "multiple-choice"
}
```
