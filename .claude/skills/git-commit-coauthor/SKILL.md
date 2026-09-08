---
name: git-commit-coauthor
description: Generate the correct git commit Co-authored-by trailers for the AI code agent and model currently in use (Claude Code, Codex, OpenCode, or ZCode with qwen, glm, minimax, deepseek, kimi, claude, or gpt models). Use when preparing or writing a git commit message, staging a commit, or attributing AI-assisted changes. Subject line is free-form in this repository (suggested `<task-dir>: <summary>`).
---

Adapted from `rstsr-agents` for this repository: subject convention relaxed to
free-form, script path fixed. Trailer rules are unchanged and mandatory.

## When to add co-authors

Add `Co-authored-by` trailers only when the AI agent materially authored or modified the committed change. Skip for trivial one-liners the user wrote.

## Commit message shape

```
<subject>

<body details>

Co-authored-by: Agent Name <Agent Email>
Co-authored-by: Model Name <Model Email>
```

- Subject is **free-form** here. Suggested shape: `<task-dir>: <summary>` for work inside a task directory, e.g. `2026-09-08-bench-sum: add rayon variant`; anything clear is fine for repo-level changes.
- Prefer details in the body.
- The two co-authors are the **agent** and the **model**, with **no blank line between** them, after one blank line following the body.

## Resolve agent and model

1. Identify the agent: `Claude Code` or `Codex` or `OpenCode` or `ZCode` (whichever is running this skill).
2. Identify the current model **with version**, e.g. `glm-5.2`, `qwen3.5-plus`:
   - Claude Code: from the `/model` property.
   - Codex: from `model` in `~/.codex/config.toml`, or the session's active model.
   - ZCode: from the session's active model.
3. Resolve emails from the registry below. **Do not guess an email.** If the agent or model is not listed, ask the user explicitly.

## Registry

Authoritative source is `scripts/coauthor.sh` in this skill. Keep this table in sync with it.

Agents:
| Agent | Email |
| --- | --- |
| Claude Code | noreply@anthropic.com |
| Codex | noreply@openai.com |
| OpenCode | support@open-code.ai |
| ZCode | service@zhipuai.cn |

Models (match by prefix, case-insensitive):
| Family | Example | Email |
| --- | --- | --- |
| `glm*` | glm-5.2 | service@zhipuai.cn |
| `deepseek*` | deepseek-v4-pro | service@deepseek.com |
| `qwen*` | qwen3.5-plus | qianwen_opensource@alibabacloud.com |
| `minimax*` | MiniMax-M2.5 | model@minimax.io |
| `kimi*` | kimi-k2.5 | growth@moonshot.cn |
| `claude*` | claude-4.6-opus | noreply@anthropic.com |
| `gpt*` | gpt-5.5 | noreply@openai.com |

## Generator

Run the script to get the exact trailer block (avoids mis-typing emails):

```bash
bash .claude/skills/git-commit-coauthor/scripts/coauthor.sh --agent "Codex" --model "glm-5.2"
```

Output (no blank line between co-authors):

```
Co-authored-by: Codex <noreply@openai.com>
Co-authored-by: glm-5.2 <service@zhipuai.cn>
```

Append it after one blank line at the end of the commit message.
