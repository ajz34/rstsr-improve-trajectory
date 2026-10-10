# GRILL-INIT-PROMPT — tensordot

Record of the prompt that started this grilling (verbatim).

---

You are going to implement tensordot in rstsr-core, with CPU-serial and rayon-auto-impl.
You are not going to implement very fast algorithms (it is known that tensordot is some
tensor contraction, which should achieve FLOPs peak to some extent; but you are not
required to do this). Make some naive implementation is okay.
Your major task is to implement and make comply to python array api compatibility.
Your similar function task is vecdot, I believe.

The most important thing is API design. You are grilled to see if you have any problems
on API design things.
For rstsr, you are going to use the workspace for this task. You can create a new
directory in rstsr-improve-trajectory for this task.
