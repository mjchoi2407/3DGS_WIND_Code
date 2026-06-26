# 2026-06-25 05 VS Code Codex history loading check

## Context

The user asked whether moving the Wind3DGS workspace from `/mnt/h/...` to WSL
internal storage would also fix slow conversation history loading in the VS
Code Codex extension.

## Findings

- The active VS Code extension is installed under WSL internal storage:
  `/home/choi/.vscode-server/extensions/openai.chatgpt-26.616.81150-linux-x64`.
- Codex state/history is also under WSL internal storage:
  `/home/choi/.codex`.
- `/home/choi/.codex` and `/home/choi/.vscode-server` are on ext4 (`/dev/sdd`).
- The project workspace is on the Windows-mounted `9p` filesystem:
  `/mnt/h/2026_paper_work/Wind_Deformable_3DGS`.
- Project-local session notes are small:
  root `sessions` 8 KB, `code/sessions` 72 KB, `ideas/sessions` 36 KB,
  `experiments/sessions` 64 KB.
- Global Codex state is larger:
  `/home/choi/.codex` 185 MB, including `sessions` 45 MB,
  `logs_2.sqlite` 32 MB, `.tmp` 87 MB, and several SQLite WAL files.
- There are 11 global Codex session files; the largest is about 31 MB:
  `/home/choi/.codex/sessions/2026/05/06/rollout-2026-05-06T11-39-16-019dfb27-629e-7040-90fd-fa4f7e6e7347.jsonl`.
- Extension logs contain many `thread-stream-state-changed` warning broadcasts,
  but no obvious direct timing entry for slow history-list loading was found.

## Conclusion

Moving the workspace to WSL internal storage should improve workspace file I/O,
git/file watcher behavior, environment startup, and extension interactions with
project files. It is unlikely to fully fix slow Codex conversation history
loading, because the extension and global conversation history are already on
WSL internal ext4 storage.

## Next

- Move the workspace for project I/O performance.
- Separately consider archiving large old global Codex session files or cleaning
  stale Codex temp/log data if conversation-list loading remains slow.
