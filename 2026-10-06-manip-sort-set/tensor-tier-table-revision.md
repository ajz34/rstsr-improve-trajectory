Manual request on tensor tier API revision.

First of principle (you should also noted in rstsr-agents code-style related skill), that
- `func_f` (fallible) should share exactly same signature as `func` (panicking) except for the return type, which is `Result<...>` instead of `...`;
- Associated method `x.func(...)` usually should share same signature from the second parameter onward, but there can be some exceptions.
- Some functions are not suitable for associated methods, example can be `rt::asarray`.
- For functions that involves two tensors, `rt::func(x1, x2, ...)` and `x1.func(x2, ...)` are usually equivalent.
- For argument overloads, if `()` is a valid argument overload, then usually you need to implement `None` also as overload. The `None` must be some of `Option<Type>`, and the type can be any proper type. Just make `None` equilvant to `()`.
- Note double parenthesis is required for overloading cases. The following list also follows this rule. For example, `rt::func(x, y)` and `rt::func((x, y))` are two different signatures; the former is overload by tuple, the latter is common two-parameter signature.

For the functions noted:
1. repeat: free fn `rt::repeat(x, (repeats, axis))`; assoc fn `x.repeat((repeats, axis))`, overloads on `RepeatArgs = (repeats, axis) / repeats`.
2. roll: free fn `rt::roll(x, (shift, axis))`; assoc fn `x.roll((shift, axis))`, overloads on `RollArgs = (shift, axis) / shift`.
3. tile: free fn `rt::tile(x, repetitions)`; assoc fn `x.tile(repetitions)`.
4. sort: free fn `rt::sort(x, (axis, descending, stable))`; assoc fn `x.sort((axis, descending, stable))`. overloads on `SortArgs = (axis, descending, stable) / (axis, descending) / axis / descending / () / None`.
5. argsort: same to function sort.
6. sort_custom: free fn `rt::sort_custom(x, sort_args, cmp)`; assoc fn `x.sort_custom(sort_args, cmp)`. overloads on `sort_args`, same to function sort.
7. argsort_custom: same to function sort_custom.
8. searchsorted: free fn `rt::searchsorted(x1, x2, (side, sorter)`; assoc fn `x1.searchsorted(x2, (side, sorter))`. overloads on `SearchSortedArgs = (side, sorter) / side / sorter / () / None`.
9. take_along_axis: free fn `rt::take_along_axis(x, indices, axis)`; assoc fn `x.take_along_axis(indices, axis)`. parameter `axis` can be set as None.
10. isin: free fn `rt::isin(x1, x2, invert)`; assoc fn `x1.isin(x2, invert)`. parameter `invert` is required field.
11. diff: free fn `rt::diff(x, axis, n, prepend, append)`; assoc fn `x.diff(axis, n, prepend, append)`. no overloads. axis/prepend/append can be None.
12-16: no change.

Mention above, `axis` can be None by TryInto AxisIndex. For other None, you may need Option.

In the above, I think there is no real need for XAPI-like traits. These traits exists for asarray, but asarray and many creation functions are themselves exceptional (creation functions does not have first argument a tensor).

During your edit on these functions, you can help me also check existing if code base does not follow the above principle. If so, not fix it, but you can report it to me. You can spawn a subagent for checking existing codebase.
