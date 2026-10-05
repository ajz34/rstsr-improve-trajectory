# Grill init prompt — 2026-10-05 reduction APIs (rstsr-core)

Verbatim prompt that started this grilling:

> You are going to enhancing reduction function implementation in rstsr-core.
>
> Sum is typical reduction function.
> Currently we have sum (no arguments), sum_axes (with axes argument), but no keepdims, and no output dtype.
> For the keepdims, you may use similar scheme in reshape_with_args, to also create sum_with_args. But for dtype, I'm currently not having a good idea.
> Also, currently we do not have any function that allow the user to input their's own customized reduction function (like that the user may wish Lp norm).
> This task may involve multiple aims:
> - Add keepdims and *_with_args
> - Try propose your thoughts on output dtype
> - Add function norm/norm_axes/norm_with_args, that first argument to be "l1/l2/Lp's p/..." (that matches scipy)
> - Add cumulative_sum/prod (matches numpy and python array-api)
> - Try fix current rstsr-faer-py test on statisical functions
>
> Efficiency is not the most concern at this stage.
>
> You are not only going to grilling me, but also check the current project status. You are suggested to spawn subagents. Pop questions after you browsed the project. Notice existance of rstsr-improve-trajectory.
