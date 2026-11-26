---
name: github-host-blocked-import-via-api
description: "On this machine github.com is unreachable (campus DNS maps it to a middlebox; every GitHub IP:22 answers a fake SSH banner and stalls KEX; HTTPS to github.com times out), so git fetch/push/ls-remote hang. api.github.com, codeload.github.com and objects.githubusercontent.com still work, which is enough to read the true remote tip and import missing commits byte-exactly with git hash-object/commit-tree."
metadata:
  type: reference
---

Observed 2026-11-26 16:00-16:30 CST while syncing this repo by rebase.

## Symptoms

- `git fetch`, `git push`, `git ls-remote` hang with no output.
- `getent hosts github.com` -> `20.247.166.64` (not a GitHub address);
  `api.github.com` -> `20.205.243.168` (correct).
- TCP:22 to *any* GitHub IP (`140.82.112.3`, `140.82.113.3`,
  `20.205.243.166`, `20.27.177.113`) answers `SSH-2.0-9f8779a8` — a
  middlebox banner, not GitHub's `babeld` — then stalls at KEX.
  `ssh.github.com:443` stalls the same way; `https://github.com` times out
  even with `curl --resolve` to a real IP.
- Reachable: `api.github.com` (HTTPS), `codeload.github.com`,
  `objects.githubusercontent.com`. Blocked: `github.com`,
  `raw.githubusercontent.com`, `gist.github.com`.

## Reading the real remote tip (no git needed)

```bash
curl -sS https://api.github.com/repos/<owner>/<repo>/branches/main   # commit.sha, dates, message
curl -sS "https://api.github.com/repos/<owner>/<repo>/compare/<old>...<new>"
curl -sS "https://api.github.com/repos/<owner>/<repo>/git/trees/<tree>?recursive=1"  # multi-MB; use urllib, curl -m can truncate
curl -sS "https://api.github.com/repos/<owner>/<repo>/git/blobs/<sha>"              # base64 content
```

Unauthenticated rate limit is 60 req/h; fetch only blobs that
`git cat-file -e <sha>` says are missing.

## Importing a missing commit exactly

1. `git hash-object -w --stdin` each blob from its base64 content; assert the
   resulting SHA equals the API blob SHA.
2. Rebuild the tree with a temporary index: `GIT_INDEX_FILE=/tmp/idx git
   read-tree <parent>`, then `git update-index --add --cacheinfo <mode> <sha>
   <path>` for each new path, then `git write-tree` — **pass the same env to
   write-tree** (a bare `git write-tree` writes the real index and silently
   produces the wrong tree).
3. `git commit-tree <tree> -p <parent>` with `GIT_AUTHOR_*`/`GIT_COMMITTER_*`
   (name, email, date incl. `+0800`) and the message plus a trailing newline.
   Candidate variants differ only in tz/message-newline; the one whose SHA
   equals the API commit SHA is the exact original.
4. `git update-ref refs/remotes/origin/main <sha>` once the SHA matches.

## Rebasing local work onto the reconstructed base

- Simplest is `git rebase <new-base>` (conflicts here were only the memory
  index; resolve by keeping remote lines, then the local block).
- A manual reparent (`git commit-tree` per commit, new parent) must also add
  the new base's files to **every** replayed commit's tree; otherwise each
  commit appears to delete them. Verify with
  `git diff-tree -r --diff-filter=D` over the replayed range.
- Before any `git push --force-with-lease=<ref>:<sha>`, pin the lease to the
  API-verified remote SHA so a stale/stale-remote push is rejected.
