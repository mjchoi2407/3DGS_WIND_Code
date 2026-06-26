# 2026-06-25 07 WSL runtime dir fix attempt

## Context

The user asked to start by fixing the `/run/user/1000` runtime directory issue
that appears to be breaking VS Code Server IPC in WSL.

## Findings

- Real unsandboxed state still shows `XDG_RUNTIME_DIR=/run/user/1000/`.
- `/run/user/1000` does not exist.
- `/run/user` is owned by `root:root`, so creating `/run/user/1000` requires
  root privileges.

## Attempted Fix

Attempted:

```bash
sudo install -d -m 700 -o choi -g choi /run/user/1000
```

This failed because `sudo` requires an interactive password prompt:

```text
sudo: a terminal is required to read the password
sudo: a password is required
```

## User Follow-Up

The user ran the root commands interactively. Verification afterward showed:

```text
XDG_RUNTIME_DIR=/run/user/1000/
drwx------ 2 choi choi ... /run/user/1000
```

So the runtime directory now exists with the correct owner and mode.

`code --status` still failed with:

```text
connect ENOENT /run/user/1000/vscode-ipc-cc715a6a-bacb-46d8-912c-ec68fcb19678.sock
```

This indicates the already-running VS Code Server is still using an old missing
IPC socket and needs a restart.

## Required User-Side Command

Run this in an interactive WSL terminal:

```bash
sudo install -d -m 700 -o "$USER" -g "$USER" "/run/user/$(id -u)"
echo "d /run/user/$(id -u) 0700 $USER $USER -" | sudo tee /etc/tmpfiles.d/wsl-user-runtime.conf
sudo systemd-tmpfiles --create /etc/tmpfiles.d/wsl-user-runtime.conf
```

Then restart VS Code Server and verify:

```bash
code --status
```

## Next

- Restart VS Code Server from the VS Code command palette, or close VS Code and
  restart WSL/VS Code from the Windows side.
- After restart, verify that `code --status` no longer fails with `vscode-ipc`
  ENOENT/EACCES.

## Post-Restart Verification

The user provided a later `code --status` output. It now prints the normal VS
Code status report instead of failing with a `/run/user/1000/vscode-ipc-*.sock`
`ENOENT` or `EACCES` error.

Current runtime directory state:

```text
drwx------ 2 choi choi ... /run/user/1000
/run/user/1000/vscode-ipc-*.sock exists
```

The latest `remoteagent.log` still contains repeated extension host reconnects
before `18:11:00`, but no reconnect entries after that point in the inspected
log tail. This suggests the runtime IPC fix and VS Code Server restart resolved
the low-level IPC/reconnect loop.

Remaining issues found in current logs are separate from the runtime directory:

- `openai.chatgpt-26.616.81150-linux-x64` and
  `james-yu.latex-workshop-10.16.1` both report `PendingMigrationError:
  navigator is now a global in nodejs` under VS Code `1.126.0`.
- The Codex extension starts successfully, but its webview still takes several
  seconds to mount and logs slow SQLite statements against the Codex logs DB.
- LaTeX Workshop PDF preview itself loads quickly, but opening TeX files still
  triggers root-file search, cache rebuilds, and AST parsing in the `/mnt/h`
  workspace.
