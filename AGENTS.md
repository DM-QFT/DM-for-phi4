# AGENTS.md

Project-specific instructions for Codex when working with Miyabi.

## Miyabi Context

- Reference manual: `references/Users_Guide_en_v1.8.pdf` (Miyabi System User's Guide v1.8, May 27, 2026).
- Current project code: `gw97`.
- Current user/work path: `/work/gw97/w97000/project/DM`.
- Login-G (`miyabi-g*.jcahpc.jp`) is `aarch64` and targets Miyabi-G/GH200 GPU jobs.
- Login-C (`miyabi-c*.jcahpc.jp`) is `x86_64` and targets Miyabi-C CPU jobs.
- The current repo normally lives on Miyabi-G. Check `hostname`, `uname -m`, and `pwd` before assuming otherwise.

## Storage

- `/home/w97000` quota is 50 GB. do not put large datasets or checkpoints there.
- `/work/gw97/w97000` is the main work area. Project quota is 5 TB per set.
- Check usage with `show_quota`.

## Environment

- On Miyabi-G, NVIDIA HPC SDK modules are loaded by default for compiler/MPI workflows.
- This project commonly uses:

  ```bash
  source /etc/profile.d/modules.sh
  module load miniforge3
  nenv2
  ```

- Use `show_module` to discover available software and `module list` in job logs for reproducibility.

## Diagnostics Checklist