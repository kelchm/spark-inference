# GLM-5.3-Flash on two DGX Sparks, our own recipe

The engine is [`deploy/glm53-kelchm`](https://github.com/kelchm/TensorFold/tree/deploy/glm53-kelchm) in kelchm/TensorFold: upstream TensorFold `fcf85f7a` with no kit, plus the commits listed in [recipe.json](recipe.json)'s `carries`, oldest first. They are:

- Robert Wojciechowski's conversation states on disk for every rank (`--learn-conversations`), from [ashhart/TensorFold#1411](https://github.com/ashhart/TensorFold/pull/1411), and laidick's message-start prompt cuts, from [ashhart/TensorFold#2404](https://github.com/ashhart/TensorFold/pull/2404). Both are open upstream and carried with their authors.
- Our own GLM CUDA prompt and decode work, not yet upstream. Several commits adapt patches from Mia's kit; each names its source in an `Adapted-from:` trailer and credits Mia as co-author, and the recipe repeats those sources as `adapted_from`.

`tools/check_recipe.py` checks that the commits above `engine.upstream` are exactly that list. [recipe.json](recipe.json) also pins base images and toolchain, which the image build uses, and the weights and drafter revisions, which it does not: they are what a deployment of this image must load, and qualification receipts name them.

## How it differs from glm53-mia

[`glm53-mia`](../glm53-mia/) follows Mia's kit: 110 imported patches on TensorFold `f350cd3e`, taken as a whole. This lane starts from newer upstream TensorFold and takes only the changes we choose, each a reviewed commit we can send upstream or drop. Weights, drafter and build pins are the same. The [Dockerfile](Dockerfile) follows glm53-mia's step for step; without a kit its kit arguments are empty and the image's `tf.kit.*` labels are empty.

## Kit notices

Mia's kit is Apache-2.0 and asks that its notice and credits accompany modified versions. This lane imports no kit, but its adapted commits take patches from it, so its engine carries the kit's `LICENSE`, `NOTICE.md` and `CREDITS.md` under `packaging/kits/glm-5.3-flash-dgx-sparks-1104/`, as glm53-mia's does, and the Dockerfile copies `packaging/kits` into the image's `/opt/tensorfold/LICENSES/kits/` whenever the engine has it. `tools/check_recipe.py` requires those files at `engine.commit` for every project a carry's `adapted_from` names. A recipe whose engine commit predates them must list the project in `notices_pending`, and `tools/new_release.py` and `tools/check_release.py` refuse to release it; the check accepts that list only while it is exactly what is missing, so it goes when the engine commit moves to one that carries the files.

## Serving profile

The profile lives in home-lab's deployment profiles, not here; set it the same on every rank. The engine commit has been checked on the Sparks with the switches below, but the lane has no release yet. After the engine switches come every other setting that decides how the lane serves, beside glm53-mia's native profile in home-lab (`glm53-mia-2026.11.07-agent-1m`) for reference. A value marked *to be qualified* or *to be decided* is not chosen yet; each is listed under [Open items](#open-items).

### Engine switches

- `TF_GLM_CHUNK_SOLO=4096 TF_GLM_PROMPT_PAIR=solo`: a prompt step takes two 4,096-row solo chunks in one pass. The engine refuses this at start if it does not fit the memory budget at the chosen `--context` and `--parallel`.
- `TF_GLM_SHARED_BUDGETS=0`: shared rounds size each stream's drafts from its last grant plus slack (see [ashhart/TensorFold#2720](https://github.com/ashhart/TensorFold/issues/2720)).
- Exact switches, qualified on the Sparks. Each replaces kernels or copies with ones that give the same bytes.
  - `TF_GLM_HC_FUSE=1`: a prompt call's `hc_post` also writes the next site's `hc_pre` partials.
  - `TF_GLM_SHARED_SLOT=1`: a prompt call's shared-expert down projection writes its MoE slot directly.
  - `TF_GLM_MM_FAST4=1`: 4-bit packed prompt matmuls build their BF16 weight pairs by subtraction, not int-to-float conversions.
  - `TF_GLM_ROUTER_FUSE=1`: a prompt call's router adds its eight K slices inside one launch, without the slices' scratch or a second launch to sum them.
  - `TF_GLM_COPIES=1`: priority copies of verified roots and context while drafting. It helped replies that re-emit their input with edits and was neutral elsewhere.
  - `TF_GLM_DENSE_PRUNE`: on by default, so leave it unset (`0` turns it off). A chunk's dense attention skips the rows sparse attention overwrites.
  - `TF_GLM_PDL_CHAIN`, `TF_GLM_KDA_STEP_WARPS`, `TF_GLM_HC_SPLIT` and `TF_GLM_HC_PART` stay unset: their defaults are the carried decode kernels.
- `TF_GLM_DRAFT_CONFIDENCE` stays unset, at its default of 0.7: lower thresholds were slower in every case measured.
- `TF_GLM_DENSE_BITS=4 TF_GLM_HEAD_BITS=8`: the dense linears packed to 4 bits and the output head to 8 at load, both lossy against the bf16 weights. This is the profile's value, pending the user's decision on the trade-off: 4-bit dense costs 0.028 nats a token of natural-text NLL against 8, while 8-bit dense costs about 18–21% of single-stream decode speed and about 2.4% of cold prompt speed. glm53-mia's profile uses 8 for both.
- `--kv-cache fp8`.
- The DFlash2 drafter from `recipe.json` with `--drafter-bits 4`.

Opt-in arithmetic: `TF_GLM_ABSORB_MMA=1` (tensor-core absorb and expand GEMMs, with a different FP32 accumulation order) and `TF_GLM_EXL3_BF16_SLOTS=1` (an EXL3 prompt chunk's MoE slot rows rounded to bf16). Both change rounding, so outputs drift from the defaults. On natural text they moved NLL by −0.0003 and +0.001 nats a token against 4-bit dense, which is small, and they stay opt-in: the profile leaves them unset. glm53-mia's profile sets the kit's `TF_GLM_ABSORB_MMA=1`.

### Capacity, scheduling and conversation reuse

| Setting | glm53-mia (`agent-1m`) | This lane |
| --- | --- | --- |
| `--parallel`, `--context` | 4 at 1,048,576 | to be qualified: 2 at 1,048,576, or 4 at a shorter context (see the known limit below) |
| `--max-tokens` | 32,768 | to be qualified |
| `--prompt-cache-gib` | 6 (the engine's default is 16) | to be qualified |
| Conversation reuse | `--learn-conversations`, with `--learn-gib 64 --learn-min-free-gib 50 --learn-daily-gib 128` | to be qualified: `--learn-conversations` or `TF_GLM_REQUEST_STARTS=1`, not both |
| `TENSORFOLD_PREFILL_SHARE` | 0.68 | to be decided by the user (see below) |
| `TENSORFOLD_PREFILL_BUDGET` | unset (no cap) | to be qualified |
| `TF_HEAT_HIGH`, `TF_HEAT_LOW` | unset (no heat bands) | to be qualified |
| `TF_GLM_DRAFT_CONFIDENCE` | unset (0.7) | unset (0.7) |
| `TF_GLM_COPIES` | unset (off) | 1 |

- **Known limit: four streams at 1M.** At `--parallel 4 --context 1048576` the engine refuses to start: the streams need 21.11 GiB against a 14.66 GiB budget, because this engine keeps fp8 KV only as a window per stream. Until it has paged or sharded fp8 KV, the profile must serve 1M context with `--parallel 2`, or four streams at a shorter context.
- **Conversation reuse.** `--learn-conversations` keeps each rank's part of a conversation's state on disk ([ashhart/TensorFold#1411](https://github.com/ashhart/TensorFold/pull/1411)); with #1411 the engine refuses `--spill-gib` and `--checkpoint-slots` for GLM, so that disk store is the only one. `TF_GLM_REQUEST_STARTS=1` cuts prompts at message starts, so a chat's next turn resumes at its history ([ashhart/TensorFold#2404](https://github.com/ashhart/TensorFold/pull/2404)). The two are mutually exclusive: the engine refuses `TF_GLM_REQUEST_STARTS=1` at start with `--learn-conversations` (`requestStartsAlone` in `cuda_native.zig`), and also with `TF_KV_SHARE` above 0.
- **Prefill share and fill budget.** At the default `TENSORFOLD_PREFILL_SHARE` of 1, a long prompt's steps owe decoding streams nothing, so while a cold prompt fills the decoders get only the turns between its steps. A share below 1 gives them time back and finishes the prompt later. Which matters more is a serving-policy choice for the user. Measured on this engine with a cold 112K prompt beside one decoding stream: at the default, the prompt took about 63 s and held the decoder to about 0.9 tokens a second meanwhile; at 0.68, the decoder kept about 6.5 tokens a second and the prompt took about 89 s. `TENSORFOLD_PREFILL_BUDGET` caps the prompt rows filled per pass while a stream waits to decode. See TensorFold's RUNBOOK, "Prompt prefill share, budget and admission policy".
- **Heat bands.** `TF_HEAT_HIGH` and `TF_HEAT_LOW`, in degrees Celsius and set together, make each prompt chunk wait while the hottest thermal zone is above the high band, until it is at or under the low one; the reply's tokens do not change. The RUNBOOK's measured example is 92 and 88 on three GB10s at TP=3.

glm53-mia's profile also sets `TENSORFOLD_MEMORY_RESERVE_GIB=12`, `TF_ROCE=1`, `TF_ROCE_MAX_KB=4096` and `TF_GLM_CLEAR_THINKING=0`, which this engine reads too; their values for this lane are to be qualified. Its `TF_KV_PAGES=on` must not be carried over: with `--kv-cache fp8` this engine refuses `on` and `all`, so leave it unset (or `auto` or `off`). Its `TF_GLM_KV_SHARD`, `TF_PROMPT_CACHE_FIRST`, `TF_GLM_HOST_CACHE_GIB`, `TF_GLM_CE_EXCHANGE` and `TF_GLM_W8A8` come from Mia's kit, and this engine ignores them. Request defaults (thinking, reasoning effort, top-k, vision, the chat template) stay with home-lab's profile.

### Open items

Before a release of this lane is qualified:

1. Capacity: `--parallel 2` at 1M context or `--parallel 4` at a shorter one, then `--max-tokens` and `--prompt-cache-gib`, with the paired 4,096-row solo chunks fitting the memory budget. Four streams at 1M wait on paged or sharded fp8 KV in the engine.
2. The user's decision on dense bits: keep 4, trading natural-text NLL for decode speed, or take 8.
3. The user's decision on `TENSORFOLD_PREFILL_SHARE`: a cold prompt's speed against concurrent decoders' speed. Then `TENSORFOLD_PREFILL_BUDGET`.
4. Conversation reuse: `--learn-conversations` with its sizes, or `TF_GLM_REQUEST_STARTS=1`.
5. Heat bands.
6. `TENSORFOLD_MEMORY_RESERVE_GIB`, `TF_KV_PAGES` and the RoCE settings.

Qualification receipts live with each release, in its `qualification.receipt`. This lane has no release yet.
