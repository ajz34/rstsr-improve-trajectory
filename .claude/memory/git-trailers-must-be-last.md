---
name: git-trailers-must-be-last
 description: Co-authored-by trailers written before the body are not trailers - `git interpret-trailers --parse` returns nothing and GitHub shows no co-author; check commit messages with that command
metadata:
  type: feedback
---

Seen in rstsr `261021/cuda` (2026-11-26): six of eleven commits put the two `Co-authored-by` lines right
after the subject, before the body; GitHub attributes nobody. The five with the block after the body, as the
last paragraph, parse correctly. Check a range with:

```sh
for c in $(git log --reverse --format=%h A..B); do
  echo "$c $(git log -1 --format=%B $c | git interpret-trailers --parse | grep -c Co-authored-by)"
done
```

(2 = both trailers recognized.) Fixing unpushed history = reword (`git rebase -i` with `exec git commit --amend
--trailer`), which is the owner's call in the rstsr repos.
