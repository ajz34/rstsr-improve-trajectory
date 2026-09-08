---
name: zcode-user-scope-skills
description: mattpocock-skills v1.2.3 installed for ZCode user scope on 2026-09-08; source of truth is the Claude Code plugin cache; how to refresh the copy.
metadata:
  type: reference
---

User-scope skills for ZCode live in `~/.zcode/skills/` (ZCode does not read
`~/.claude/`). On 2026-09-08 all 25 skills of the `mattpocock-skills` plugin
v1.2.3 were copied there, flattened (one directory per skill, e.g.
`~/.zcode/skills/grilling/SKILL.md`).

- **Source of truth**: Claude Code plugin cache at
  `~/.claude/plugins/cache/claude-plugins-official/mattpocock-skills/<version>/`
  (enabled in `~/.claude/settings.json`). Updates via Claude Code's marketplace
  do **not** propagate to the ZCode copy.
- **Refresh procedure**: re-copy the skills listed in the plugin's
  `.claude-plugin/plugin.json` (`skills` array) into `~/.zcode/skills/`,
  flattening `skills/<category>/<name>/` → `~/.zcode/skills/<name>/`. Each
  skill dir is self-contained (SKILL.md + support files + `agents/`);
  `unregister`-style skills in `deprecated/`, `in-progress/`, `misc/` are
  intentionally not installed (not in the manifest).
- `grill-me` is a one-line wrapper that runs a `grilling` session; the useful
  logic is in `grilling`/`grill-with-docs`.
- ZCode picks up new user-scope skills on session start; a running session may
  need a restart/reload to list them.
