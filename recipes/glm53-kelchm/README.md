# GLM-5.3-Flash on two DGX Sparks, our own recipe

The engine is [`deploy/glm53-kelchm`](https://github.com/kelchm/TensorFold/tree/deploy/glm53-kelchm) in kelchm/TensorFold: upstream TensorFold `v1.3.6` (`a681f003`) with no kit, plus the 22 commits listed in [recipe.json](recipe.json)'s `carries`, oldest first. They port paged FP8 KV and cross-page disk save/restore onto `v1.3.6`, persist unsalted image conversation metadata in the rows disk store, allow `TF_GLM_REQUEST_STARTS=1` alongside `--learn-conversations` via format-2 disk headers carrying the `below` cut digest, and add our verified bit-exact P5 GLM CUDA prompt, decode, DFlash2, and TP2 RoCE optimizations.

`tools/check_recipe.py` checks that the commits above `engine.upstream` are exactly that list. [recipe.json](recipe.json) also pins base images and toolchain (`zig 0.17.0`, `sm=121`, `libjpeg-turbo 3.1.4.1`, `libwebp 1.6.0`, `xgrammar v0.2.8`), which the image build uses, and the weights and drafter revisions, which it does not: they are what a deployment of this image must load, and qualification receipts name them.

## How it differs from glm53-mia

[`glm53-mia`](../glm53-mia/) follows Mia's kit: 110 imported patches on TensorFold `f350cd3e`, taken as a whole. This lane starts from upstream TensorFold `v1.3.6` (`a681f003`) and takes only our reviewed, upstreamable commits. Weights, drafter and build pins are the same. The [Dockerfile](Dockerfile) follows glm53-mia's step for step while baking the shared [runtime/](../../runtime/) contract and supporting optional prebuilt fatbins; without a kit its kit arguments are empty and the image's `tf.kit.*` labels are empty.

## Kit notices

The engine commit carries the verbatim Apache-2.0 `LICENSE`, `NOTICE.md` and `CREDITS.md` from `MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104` under `packaging/kits/glm-5.3-flash-dgx-sparks-1104/`, and the [Dockerfile](Dockerfile) copies `packaging/kits` into `/opt/tensorfold/LICENSES/kits/` in the image. `tools/check_recipe.py` verifies `packaging/kits` whenever a carry adapts a kit's patches or a lane imports a kit.

## Serving profile

The profile lives in home-lab's deployment profiles, not here; set it the same on every rank. This `v1.3.6` P5 engine commit has been qualified on the two GB10 DGX Sparks with the bit-exact serving switches below; the table shows them beside `glm53-mia`'s reference native profile in home-lab (`glm53-mia-2026.11.07-agent-1m`).

### Verified bit-exact P5 switches

- Quantization and KV layout: `TF_GLM_DENSE_BITS=4`, `TF_GLM_HEAD_BITS=8`, `TF_KV_PAGES=on`, `--kv-cache fp8`, and the pinned DFlash2 drafter with `--drafter-bits 4`. Paged FP8 KV unlocks four concurrent 1M-context streams on 2× GB10, allocating `5,413` pages (`2,771,456` active KV tokens at 512 rows/page).
- Bit-exact prompt, decode, DFlash2 and RoCE switches (qualified on GB10):
  - `TF_GLM_DFLASH_ONLY=1`: routes speculative drafting strictly through DFlash2 with selector vocab bounds checking.
  - `TF_GLM_DFLASH_Q4_DOWN=1`: uses Q4 MLP down projections (with wide 32/64/128-row Q4 tiles) in DFlash2 and the draft head.
  - `TF_GLM_COPY_8_16=1`: verified 8/16-token context copy policy during drafting.
  - `TF_GLM_EXL3_DECODE=1` and `TF_GLM_EXL3_N_MAJOR=1`: EXL3 decode kernels across 1..32 rows with N-major layout on rows 4 and 8..32.
  - `TF_GLM_PACKED_PROMPT_S2=1`: bit-exact two-stage packed prompt projections with L2 row-block groups.
  - `TF_GLM_DFLASH_SNAP_CONTEXT=1`: 640-row DFlash2 prompt context snapshot overlay.
  - `TF_GLM_PAIR_DFLASH=1`: pairs prompt steps beside DFlash2 rounds (clearing `TF_GLM_REQUEST_STARTS` cut boundaries).
  - `TF_GLM_MULTI_PREFILL=1`: batches short prompts beside stepped long-context fills.
  - `TF_GLM_ROCE_FUSE=1`: TP2 RoCE post-epilogue fusion (paired with `TF_ROCE=1` and `TF_ROCE_MAX_KB=4096`).
  - `TF_GLM_REQUEST_STARTS=1`: message-start prompt cuts, now compatible with `--learn-conversations` via format-2 disk headers carrying the `below` cut digest.
  - `TF_GLM_CLEAR_THINKING=0`: preserves prior turn reasoning across multi-turn replays.
- Explicitly kept off: `TF_GLM_L2PF_SHARED_FIRST=0` and `TF_GLM_PRICED_WINDOWS=0` stay disabled because live GB10 A/B measurements showed regressions with either enabled.

### Capacity, scheduling and conversation reuse

| Setting | glm53-mia (`agent-1m`) | This lane (`v1.3.6` P5) |
| --- | --- | --- |
| `--parallel`, `--context` | 4 at 1,048,576 | 4 at 1,048,576 (`TF_KV_PAGES=on --kv-cache fp8`, `5,413` pages = `2,771,456` KV tokens on 2× GB10) |
| `--max-tokens` | 32,768 | 32,768 |
| `--prompt-cache-gib` | 6 | 8 |
| Conversation reuse | `--learn-conversations` (`--learn-gib 64 --learn-min-free-gib 50 --learn-daily-gib 128`) | `--learn-conversations` (`--learn-gib 64 --learn-min-free-gib 50 --learn-daily-gib 128`) together with `TF_GLM_REQUEST_STARTS=1` |
| `TENSORFOLD_PREFILL_SHARE` | 0.68 | 0.68 |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | 12 | 12 |
| `TF_ROCE`, `TF_ROCE_MAX_KB` | 1, 4096 | 1, 4096 (with `TF_GLM_ROCE_FUSE=1`) |
| `TF_GLM_CLEAR_THINKING` | 0 | 0 |
| `TF_GLM_L2PF_SHARED_FIRST`, `TF_GLM_PRICED_WINDOWS` | n/a | 0, 0 (off; GB10 A/B regressions) |

- **Four streams at 1M context.** With paged FP8 KV (`TF_KV_PAGES=on --kv-cache fp8`), `--parallel 4 --context 1048576` starts cleanly on 2× GB10 and yields `5,413` pages (`2,771,456` active KV tokens), comfortably above the `2,582,528`-token `4×1M` capacity floor.
- **Conversation reuse and image persistence.** `--learn-conversations` keeps each rank's slice of a conversation's state on disk and now works alongside `TF_GLM_REQUEST_STARTS=1` (storing the `below` cut digest in format-2 disk headers so message-start cuts and cross-restart disk reads compose cleanly). Unsalted image conversations also persist their image metadata in the rows disk store across restarts. `TF_KV_SHARE > 0` remains incompatible with both `--learn-conversations` and `TF_GLM_REQUEST_STARTS=1`.
- **Prefill share and memory reserve.** `TENSORFOLD_PREFILL_SHARE=0.68` protects concurrent decoding streams while a cold prompt fills, and `TENSORFOLD_MEMORY_RESERVE_GIB=12` leaves headroom for unified memory and disk-save staging. `TF_HEAT_HIGH` and `TF_HEAT_LOW` remain available if thermal throttling bands are needed.

Request defaults (thinking, reasoning effort, top-k, vision, and the chat template) stay with home-lab's profile. Mia-kit-only env vars (`TF_GLM_KV_SHARD`, `TF_PROMPT_CACHE_FIRST`, `TF_GLM_HOST_CACHE_GIB`, `TF_GLM_CE_EXCHANGE`, `TF_GLM_W8A8`) are ignored by this engine.

### Container runtime contract

This lane's [Dockerfile](Dockerfile) bakes the shared [runtime/](../../runtime/) package at `/opt/spark-inference/runtime/` with `/usr/local/bin/tensorfold-entrypoint` (default `CMD`) and `/usr/local/bin/tensorfold-health`. See [runtime/README.md](../../runtime/README.md) for the full contract:

- **3-way mounts**: HF model cache (`HF_HOME`), runtime inputs/drafter/template (`TF_RUNTIME_INPUTS_DIR`, `DRAFTER`, `CHAT_TEMPLATE`), and dedicated persistent conversation store (`GLM_LEARN_DIR` / `GLM_LEARN_ROOT`).
- **Orchestrator-neutral TP & RDMA**: accepts SparkRun (`RANK`, `WORLD_SIZE`, `FABRIC_MASTER`, `MASTER_PORT`, `NCCL_IB_HCA`) and Kubernetes LeaderWorkerSet (`LWS_WORKER_INDEX`, `LWS_GROUP_SIZE`, `LWS_LEADER_ADDRESS`, `TF_ROCE_HCA`) with zero hardcoded host interfaces, so the same image runs unchanged on DGX OS and Talos.
- **PID-1 supervision & Rank-1 `SIGTERM` flush**: rank 0 forwards `SIGTERM` immediately for HTTP drain + disk flush; rank 1 holds off signaling the follower engine for `RANK1_FLUSH_GRACE_SECONDS` (default `60` s, with `>=75` s outer stop grace) so rank 0 can gather rank 1's pages over the TP link and finish writing `--learn-conversations` states before follower exit.
- **MemGuard & readiness probes**: unified-memory accounting (`MemAvailable` + parked per-CPU pages, plus finite ancestor cgroup v2 caps) and redirect-refusing, secret-redacting HTTP + worker state health probes.

## Fast GB10 & prebuilt fatbin builds

By default, CI and `tools/dev_build.sh` compile CUDA kernels only for GB10 (`-Dsm=121`). For fast local rebuilds against `v1.3.6+` when CUDA fatbins have already been compiled, pass prebuilt artifacts to `tools/dev_build.sh`:

```sh
FATBINS_DIR=/path/to/fatbins TP_FATBIN=/path/to/tp_mailbox.fatbin TOOLS=0 \
  tools/dev_build.sh <engine-commit> glm53-kelchm
```

This mounts `prebuilt-fatbins` into the `build` stage and passes `-Dfatbins` / `-Dtp-fatbin` to `zig build`. Set `TP_NVCC=1` when compiling the TP mailbox kernel from source without a prebuilt `tp_mailbox.fatbin`.

## Cutting a release

Once this `v1.3.6` P5 recipe merges to `main` and CI publishes `ghcr.io/kelchm/spark-inference/glm53-kelchm:sha-<commit>` with its build provenance attestation:

1. Run `python3 tools/new_release.py glm53-kelchm` to write `recipes/glm53-kelchm/releases/glm53-kelchm-<date>.json` and refresh the README status table.
2. Create and push the printed `release/<release>` tag on `e0fd8b1eb33e07d55afeb2e3bb2a5700212b9d06` in `kelchm/TensorFold`, then verify with `python3 tools/check_release.py`.
3. Record home-lab's qualification receipt on the release file per [AGENTS.md](../../AGENTS.md) after deploying and running the expanded [qualification/](../../qualification/) suite against the published image digest.

Qualification receipts live with each release, in its `qualification.receipt`. This lane has no release yet.
