# Shared Container Runtime Contract (`runtime/`)

`runtime/` is a zero-dependency Python stdlib package baked into `recipes/glm53-kelchm` images at `/opt/spark-inference/runtime/` with executable shims at:

- `/usr/local/bin/tensorfold-entrypoint` (`python3 -m runtime.entrypoint`)
- `/usr/local/bin/tensorfold-health` (`python3 -m runtime.health`)

The exact same image runs unchanged under **SparkRun/Docker on DGX OS** and **Kubernetes LeaderWorkerSet + CDI + RDMA on Talos** because all host mounts, RDMA devices, network interfaces, and rank rendezvous coordinates are parameterized through environment variables.

## 3-way mount contract

| Mount | Default container path | Env vars | Ownership |
| --- | --- | --- | --- |
| 1. HF model cache | `/cache/huggingface` | `HF_HOME`, `MODEL_PATH`, `HF_HUB_OFFLINE=1` | Read-only weights cache shared across runs |
| 2. Runtime inputs / drafter | `/opt/tensorfold-runtime/inputs` | `TF_RUNTIME_INPUTS_DIR`, `DRAFTER`, `CHAT_TEMPLATE` | Optional custom `chat_template.jinja` and pinned DFlash2 drafter |
| 3. Persistent conversation store | `/var/lib/tensorfold/learn/<profile>` (or `/cache/huggingface/spark-inference/learn/<profile>` on SparkRun) | `GLM_LEARN_DIR`, `GLM_LEARN_ROOT`, `GLM_LEARN_GIB`, `GLM_LEARN_MIN_FREE_GIB`, `GLM_LEARN_DAILY_GIB` | Dedicated writable NVMe directory on rank 0; refuses `..`, `/`, `HF_HOME`, and legacy `tensorfold-spill` paths |

## Parameterized RDMA & TP rendezvous

No host interface (`enp1s0f0np0`, `enP7s7`) or HCA name (`rocep1s0f0`, `mlx5_0`) is hardcoded.

- **Rank / world**: reads `RANK` / `WORLD_SIZE` (SparkRun), `LWS_WORKER_INDEX` / `LWS_GROUP_SIZE` (Kubernetes LeaderWorkerSet), or `TF_GLM_RANK` / `TF_GLM_WORLD`.
- **Master address**: reads `FABRIC_MASTER`, `LWS_LEADER_ADDRESS`, `MASTER_ADDR`, or `TF_TP_MASTER`. Accepts both IPv4 literals and RFC-1123 hostnames (resolved to IPv4 for `NCCL_COMM_ID` / `tp_watch`).
- **Bootstrap port**: `MASTER_PORT` + `TF_TP_PORT_OFFSET` (default `-1` so `tp_watch` binds `MASTER_PORT` on SparkRun; set `0` if passing the NCCL port directly). Sets `--master`, `--master-port`, and matching `NCCL_COMM_ID` so TensorFold's `rendezvous.fromMaster` never trips `RendezvousDisagree`.
- **RDMA HCAs**: reads `TF_ROCE_HCA` or `NCCL_IB_HCA`, parses up to two active devices (stripping optional `=` and `:1`), refuses `^` exclusion lists, verifies `/sys/class/infiniband/<hca>` when present, and passes through `TF_ROCE_GID_INDEX`/`NCCL_IB_GID_INDEX`, `TF_ROCE_MAX_KB` (default `4096`), and optional `NCCL_SOCKET_IFNAME`.

## PID-1 lifecycle & Rank-1 `SIGTERM` flush

- **Rank 0 (head)**: forwards `SIGTERM`/`SIGINT` immediately to the engine's process group so `serve.zig` switches `/health` to `503 draining`, drains active requests, gathers rank 1's conversation slices over the TP link, flushes pending `--learn-conversations` disk writes, and sends the clean stop signal to followers. A second stop signal is forwarded so `serve.zig` cuts drain short. Waits up to `SHUTDOWN_GRACE_SECONDS` (default `45` s) before `SIGKILL` of the process group.
- **Rank 1+ (worker)**: follower ranks run `follow.follow()` without `serve.zig`'s HTTP drain handler. When an orchestrator sends `SIGTERM` to both ranks simultaneously, killing rank 1 immediately would break rank 0's mid-flush disk gather. Instead, rank 1's entrypoint uses clean TP shutdown as the completion signal and holds off forwarding `SIGTERM` for `RANK1_FLUSH_GRACE_SECONDS` (default `60` s, strictly exceeding rank 0's `45` s grace plus signal delivery skew) so rank 0 finishes flushing and ends rank 1 naturally (exit `0`), then syncs filesystem buffers. A second stop signal cuts the wait short; if the flush window expires, it sends `SIGTERM` and waits `RANK1_KILL_GRACE_SECONDS` (default `10` s) before `SIGKILL`, preserving the child's actual exit code. Configure outer Docker `stop_grace_period` / Kubernetes `terminationGracePeriodSeconds` to at least `75` s.
- **SIGHUP**: forwarded to the child process group for live API key reload.
- **Zombie reaping & group teardown**: PID 1 reaps orphaned grandchildren without losing the tracked engine child's exit code and always kills any remaining descendants in the private process group when the leader exits.
- **Engine compatibility**: `TF_KV_PAGES=on` with `--kv-cache fp8` and `--parallel 4 --context 1048576` targets the `v1.3.6` paged-FP8 engine pinned by `recipes/glm53-kelchm` (`e0fd8b1e`); earlier pre-`v1.3.6` commits such as `01b5396a` require `TF_KV_PAGES=auto` (or unset) and `--parallel 2` at 1M context. Engine P3/P5 env knobs (`TF_GLM_MULTI_PREFILL`, `TF_GLM_ROCE_FUSE`, `TF_GLM_PRICED_WINDOWS`, `TF_GLM_REQUEST_STARTS`, `TF_GLM_DFLASH_*`, `TF_GLM_EXL3_*`, `TF_GLM_PACKED_PROMPT_S2`, `TF_GLM_PAIR_DFLASH`, `TF_GLM_COPY_8_16`) pass through unmodified (`TF_GLM_REQUEST_STARTS=1` is permitted with `--learn-conversations`; `TF_KV_SHARE > 0` remains mutually exclusive with both). Optional `VISION_VIDEO_TOKENS` (`64..262144`) passes `--vision-video-tokens`.

## MemGuard (DGX Spark unified memory)

Samples `/proc/meminfo` (`MemAvailable`, `MemTotal`) before `/proc/zoneinfo` (per-CPU parked pages above `high_min`, 512 MiB fallback), caps at `MemTotal`, and takes the minimum with finite cgroup v2 `memory.max - memory.current` when present. Sustaining below `MEMGUARD_FLOOR_GIB` (default `2.5` GiB) for `MEMGUARD_SUSTAIN_SECONDS` (default `4` s) stops the rank cleanly and exits `73`; unreadable telemetry exits `72`. Applies `MEMGUARD_OOM_SCORE_ADJ` (default `1000`) to the engine child.

## Readiness & worker probes

- Head readiness: `tensorfold-health --mode ready --url http://127.0.0.1:8888 --model GLM-5.3-Flash --lanes 4` (add `--log /var/log/tensorfold/rank0.log --min-pool-tokens 2582528` to also gate on the `4×1M` paged KV pool floor when native `/health` does not report `pool_tokens` / `kv_shard`).
- Head liveness: `tensorfold-health --mode live --url http://127.0.0.1:8888`
- Worker (rank 1) readiness / liveness: `tensorfold-health --mode worker` / `--mode worker-live` (checks `/tmp/tensorfold-runtime-state.json` and child PID liveness without requiring an HTTP port on rank 1).
- Secrets (`TENSORFOLD_API_KEY`, bearer headers) are read from `--api-key-env` and redacted from every error message.
