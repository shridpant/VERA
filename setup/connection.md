# Connection record (identifiers only; no secrets)

Verified 2026-10-22 by the setup agent (non-interactive, key-based SSH from the user's Mac).

| Item | Value |
|---|---|
| Oz login ID | `sp2117` |
| SSH login host | `oz.gccis.rit.edu` (`hostname -f` → `oz.gccis.rit.edu`) |
| Working SSH command | `ssh sp2117@oz.gccis.rit.edu` (also works with `-o BatchMode=yes`) |
| Jump host / VPN | None needed from the user's Mac on 2026-10-22. Whether off-campus access needs the RIT VPN was not tested. |
| GPU execution route | Oz itself is the GPU host. No Slurm/PBS/HTCondor/LSF client exists. Jobs run detached on the login host with `setup/scripts/launch.sh`, which pins one idle GPU by UUID. |
| OS | Debian GNU/Linux 13 (trixie), kernel 6.12.113+deb13-amd64 |
| GPUs (2026-10-22) | 4 × NVIDIA RTX 6000 Ada Generation (46068 MiB), 8 × NVIDIA A16 (15356 MiB), driver 615.71.10. All idle at preflight. The handoff's "2 × A100 24 GB" description does not match this host. |
| CPU / RAM | 128 CPUs, 503 GiB RAM |
| `VERA_PROJECT_ROOT` | `/home/stu11/s5/sp2117/vera_research` (`~/vera_research`, NFS home `rohan:/dcspool/home2/stu11`) |
| Environment | `/home/stu11/s5/sp2117/vera_research/setup/environment/env` (conda prefix, Python 3.10) |
| Home quota | 150 GB. The department dump `/usr/local/pub/jmc/home2_quota` read `sp2117 150G 140G` at 01:50 on 2026-10-22; `quota -s` prints nothing. |

Site rules recorded by earlier projects on this account: shared server, no sudo, no GPU policy found. Never kill other
users' processes or search their directories.
