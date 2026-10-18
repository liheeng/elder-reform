# AGENTS.md — 本仓库的 agent 操作约束

> 给所有在此仓库工作的 AI agent（含 Copilot / Claude Code）的硬性约束。
> 与本文件冲突的默认习惯，一律以本文件为准。

## 🚫 禁止触发 GitHub CI（2026-10-18 起，长期有效）

本机 GitHub Actions / CI 的免费额度已用尽。**不要做任何会向 GitHub 写入、
从而可能引发 workflow 的操作。**

**禁止**

- `git push` —— 任何 remote、任何分支都不例外
- `gh workflow run`、`gh run rerun`、`gh run *`
- 创建或修改 PR / issue（`gh pr create`、`gh issue create`、`gh pr comment` 等）
- 其他会写入 GitHub 的操作（release、tag push、wiki、gist 等）

**允许（工作流程照常，只是不出本机）**

- `git commit` —— 所有改动提交到本地
- 本机跑测试 / 训练 / 端到端回归 / 评测
- 读写文件
- 只读 git 命令：`git log` / `git diff` / `git status` / `git show` / `git ls-tree`

**例外通道**：若某项任务确实必须 push 或写远端，**先停下来问用户**，
拿到明确同意后才执行；不要自行判断"应该没事"。

## 仓库现状要点（便于判断上下文，非操作许可）

- **本地有大量未推送提交**（截至 2026-10-18 为 26 个，后续本地提交会继续增加）
- `origin`(liheeng/elder-reform) 与 `upstream`(yyx356/elder-reform) **都没有
  `.github/workflows/`** —— 即本仓库本身未配 Actions。如果 CI 额度仍在被消耗，
  原因不在这个仓库，需要另行排查。
- 生产模型：`models/lstm_v2_nonperson.pt`（`models/` 与 `*.pt` 被 gitignore）
- **旧数据集唯一副本在 `processed/_baseline_noscore/`**（`*.npy` 被 gitignore）。
  它无法由当前代码复现（差 `conf` 与质量门两个变量），是历史对比的唯一依据，**不要删**。
- `processed/*_olddata.npy` 是指向上述副本的软链，供 `train_v2.py --suffix _olddata` 使用。
