# AGENTS.md — 本仓库的 agent 操作约束

> 给所有在此仓库工作的 AI agent（含 Copilot / Claude Code）的硬性约束。
> 与本文件冲突的默认习惯，一律以本文件为准。

## ✅ 要 push；🚫 但不要触发 CI（2026-10-18 起，长期有效）

本机 GitHub Actions / CI 的免费额度已用尽。

**要做的：代码改动照常提交，并 push 到远端。** 不要只留在本地。

**不要做的：任何会触发 CI / workflow 的操作。**

- `gh workflow run`、`gh run rerun`、`gh run *`
- 手动 dispatch 任何 workflow
- 往 PR 分支 push（**`pull_request` 事件同样会触发 CI**）
- release / tag push（可能触发 release workflow）

## 当前事实：本仓库 push 本身不会跑 CI

已核实（2026-10-18，只读检查）：

| 检查项 | 结果 |
|---|---|
| 本地 `.github/` 目录 | 不存在 |
| `origin/main` 的 `.github/workflows/` | **0 个** |
| `upstream/main` 的 `.github/workflows/` | **0 个** |
| GitLab / Travis / Azure / CircleCI / Jenkinsfile | 均无 |

**→ 现在 `git push` 到 `origin` 不会触发任何 CI，可以放心 push。**
（若 CI 额度仍在被消耗，原因不在这个仓库，需另行排查。）

## 如果将来仓库里出现了 workflow

那时 push 就会触发 CI，按优先级处理：

1. **先确认是否存在**：`git ls-tree -r --name-only origin/main -- .github/workflows`
2. **尽量用跳过标记**：commit message 带 `[no ci]`（GitHub Actions 识别的跳过关键词，
   对 `push` / `pull_request` 事件生效）。但这只是**尽力而为** ——
   workflow 若未尊重该标记，仍会跑
3. **拿不准就先问**：看不了 workflow 内容、或它明显会消耗额度时，停下来问用户

## 其他仍然要先问用户的远端操作

`git push` 已获许可，但以下操作会改变远端状态或通知他人，执行前先问：

- 建 / 改 PR、issue，或在其下评论
- 建 release、tag、删分支
- 改 remote 配置、force push、rebase 已推送的历史

## 仓库现状要点（便于判断上下文，非操作许可）

- `origin` = liheeng/elder-reform，`upstream` = yyx356/elder-reform
- 生产模型：`models/lstm_v2_nonperson.pt`（`models/` 与 `*.pt` 被 gitignore）
- **旧数据集唯一副本在 `processed/_baseline_noscore/`**（`*.npy` 被 gitignore）。
  它无法由当前代码复现（差 `conf` 与质量门两个变量），是历史对比的唯一依据，**不要删**。
- `processed/*_olddata.npy` 是指向上述副本的软链，供 `train_v2.py --suffix _olddata` 使用。
