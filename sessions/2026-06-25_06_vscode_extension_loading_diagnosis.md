# 2026-06-25 06 VS Code extension loading diagnosis

## Context

The user reported that selecting PDFs in VS Code is slow and that extension
loading generally feels unhealthy in the WSL2 + VS Code + Codex setup.

## Findings

- Workspace is on Windows-mounted `9p` storage:
  `/mnt/h/2026_paper_work/Wind_Deformable_3DGS`.
- VS Code Server, extensions, and Codex global state are on WSL ext4:
  `/home/choi/.vscode-server` and `/home/choi/.codex`.
- Installed WSL-side extensions are limited:
  `openai.chatgpt-26.616.81150-linux-x64`,
  `james-yu.latex-workshop-10.16.1`, and two Korean language-pack versions.
- PDF files under `ideas/` and `paper/` are small, with the largest checked PDF
  about 260 KB. Reading all workspace PDFs with `sha256sum` took about 0.56 s,
  so PDF file size alone is not the likely cause.
- VS Code Server logs show `XDG_RUNTIME_DIR=/run/user/1000/`, but the directory
  does not exist in the real WSL environment.
- VS Code Server and extension host fail to create/connect IPC sockets under
  `/run/user/1000/vscode-ipc-*.sock`:
  `listen EACCES: permission denied` and `connect ENOENT`.
- `code --status` fails with:
  `Unable to connect to VS Code server: connect ENOENT /run/user/1000/vscode-ipc-...sock`.
- `remoteagent.log` shows repeated extension host reconnections roughly every
  20 seconds.
- `remoteexthost.log` shows OpenAI/Codex extension activation followed by
  `PendingMigrationError: navigator is now a global in nodejs`.
- `remoteexthost.log` shows LaTeX Workshop activating on
  `onCustomEditor:latex-workshop-pdf-hook`, followed by repeated
  `TypeError: Cannot read properties of undefined (reading 'setNoDelay')`.
- LaTeX Workshop log shows PDF viewer activation sequence:
  - extension activation at `17:08:29`
  - viewer page loaded at `17:08:37`
  - root-file search completed around `17:08:59`
- Codex log shows additional startup/listing overhead:
  - app routes mounted after about 5 s
  - one `logs` DB query exceeded the 1 s threshold and took about 4.8 s
  - repeated `thread-stream-state-changed` broadcast warnings
- Git extension initial scan opened four nested repositories and multiple
  `git status -z -uall` calls took about 2.2-2.7 s on `/mnt/h`.

## Assessment

- Primary system-level issue: missing `/run/user/1000` despite
  `XDG_RUNTIME_DIR` pointing there. This breaks VS Code IPC and likely causes
  broad extension host reconnection/loading symptoms.
- Secondary workspace issue: project on `/mnt/h` makes Git scans, watchers, and
  LaTeX root discovery slower than they would be on WSL ext4.
- PDF-specific issue: LaTeX Workshop activates and performs root-file discovery
  when opening PDFs; this adds noticeable latency even though the PDFs are
  small.
- Codex-specific issue: global Codex state/log queries and the current OpenAI
  extension activation error add startup overhead independent of the project
  folder location.

## Recommended Fix Order

1. Fix the WSL runtime directory / VS Code IPC problem.
2. Fully restart VS Code Server and reconnect from Windows VS Code.
3. Move workspace source to WSL internal storage to reduce Git/watcher/LaTeX
   scan latency.
4. If PDF opening remains slow, tune LaTeX Workshop settings for this workspace.
5. If Codex history remains slow, clean or archive large Codex global state and
   temp/log files separately.

## Next

- Create `/run/user/1000` with correct ownership and mode or otherwise ensure
  `XDG_RUNTIME_DIR` points to an existing user-owned runtime directory before
  VS Code Server starts.
- After restarting VS Code Server, rerun `code --status` and confirm it no
  longer fails with `vscode-ipc` ENOENT/EACCES.
