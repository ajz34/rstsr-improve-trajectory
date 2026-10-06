Q1/2: Okay.

Q1/2 additional: It is a common technique that for col-major default layout, you can first reverse all axes of the input tensor, then follow row-major's broadcast rule and computation, then finally reverse the output tensor's axes. I'm not sure if this is applicable to current functions. And you can browse the project; if I'm right, this technique usually implemented in tier-2 (intermediate between tensor layer and real implementation; the real-implementation usually be better performance when row-major).

Q3: Okay, as long as this will pass array api tests.

Q4/5/6: Okay.

Q7: Most traits in rstsr-dtype-traits starts with `Ext*`. I'm not sure if you feel `ExtSortCmpAPI` is good. And if this can be introduced in ExtNumAPI or ExtRealAPI or something else. Anyway, this should be implemented for both integers and real floats.

Q8: Some advanced notices:
- For model selection, you are usign claude-code-router and can dispatch different models.
  - Main session use glm/glm-5.3 (Custom Opus model) with effort "high".
  - Implementation subagents use glm/glm-5.3-flash (Custom Sonnet model), with effort "max".
  - Code review use DeepSeek/deepseek-flash (Custom Haiku model), with effort "high", with some exception you use "max".
  - Code review only once, do not loop implement-review. Also, not all code review comments need to be implemented, but they are important suggestions, and if some review you feel not implemented is better, notice and address them.
  - After all code finishes, you are going to ask DeepSeek/deepseek-flash 
- You've given approximately 9 steps. For each step, you are going to do code review with effort "high".
- After everything finishes, you are going to ask DeepSeek/deepseek-flash to do a final code review with effort "max".

Q9: I think it is not manipulation, since this involves real computation. I prefer it to be tensor/diff.rs at current time. This function can be simply `pub fn diff(x, axis: isize, n: usize, prepend: Option<&TensorAny>, append: Option<&TensorAny>) -> TensorAny`, not using overloading, unlike that of sort functions.
