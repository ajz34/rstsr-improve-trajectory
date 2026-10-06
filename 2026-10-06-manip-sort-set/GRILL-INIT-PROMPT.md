# Grill init prompt — manipulation / sorting / set wave (2026-10-06)

Task under grill (verbatim from the user, /mattpocock-skills:grill-me):

> You are going to implement missing manipulation functions that related to
> python array api tests (mostly repeat, roll, tile), and sorting functions
> (mostly `sort`, `argsort`), and searching/set functions. See also
> rstsr-improve-trajectory/2026-10-04-rstsr-faer-py/reports/STATUS-2026-10-06-failure-census.md.
> For these functions, you are allowed to generate a completely new tensor
> (so always inputs &TensorAny or TensorView, or implement as
> TensorViewAPI), to make things easy.
> I think with current codebase, many things are clear. But you can also
> grill me if anything not clear. And also, though current codebase is large,
> some of the functions may have different and not-previously-existed
> signatures that you need to handle.
> You are also going to try to improve test coverage of array api tests on
> rstsr-faer-py. Still notice that rstsr-faer-py should be interface only,
> not implementing algorithms and real implementations.
> Also notice that some functions may need no device-dependent
> implementation, but some may need to. rstsr-core need to separate the
> implementation (cpu-serial, faer, possibly future non-cpu devices but
> currently we don't have) and its tensor interface.
> Note you should first spawn agents for comprehensing codebase, including
> knowing code style of existing fuctions. And when subagents converges, then
> you propose grilling questions. Do not grill too early.
> Notice that implementation and interface should be strictly separated.
> Notice rstsr-improve-trajectory. CLAUDE.md there have rule of grilling.

## Measured scope (census stamp 20261006-183350, 1014/286/82)

In-scope missing names (41 failing nodeids; G-001, G-002, G-003, G-029):

- manipulation: `repeat`, `roll`, `tile` (G-002/G-003)
- sorting: `sort`, `argsort` (G-001)
- searching/set/indexing: `nonzero`, `searchsorted` (+scalars variant),
  `isin`, `unique_all`, `unique_counts`, `unique_inverse`, `unique_values`,
  `take_along_axis` (G-029 remainder)

13 has_names + 13 signatures + 15 runtime tests.

Comprehension agents dispatched (4): rstsr-core manipulation layout,
rstsr-core searching/sorting layout + device-impl separation, rstsr-faer-py
binding patterns, array-api spec/test requirements.
