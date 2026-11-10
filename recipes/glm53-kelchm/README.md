# GLM-5.3-Flash on two DGX Sparks, our own recipe

The engine is [`deploy/glm53-kelchm`](https://github.com/kelchm/TensorFold/tree/deploy/glm53-kelchm) in kelchm/TensorFold: upstream TensorFold `fcf85f7a` with no kit, plus the commits listed in [recipe.json](recipe.json)'s `carries`, oldest first. They are:

- Robert Wojciechowski's conversation states on disk for every rank (`--learn-conversations`), from [ashhart/TensorFold#1411](https://github.com/ashhart/TensorFold/pull/1411), and laidick's message-start prompt cuts, from [ashhart/TensorFold#2404](https://github.com/ashhart/TensorFold/pull/2404). Both are open upstream and carried with their authors.
- Our own GLM CUDA prompt and decode work, not yet upstream. Several commits adapt patches from Mia's kit; each names its source in an `Adapted-from:` trailer and credits Mia as co-author, and the recipe repeats those sources as `adapted_from`.

`tools/check_recipe.py` checks that the commits above `engine.upstream` are exactly that list. [recipe.json](recipe.json) also pins base images and toolchain, which the image build uses, and the weights and drafter revisions, which it does not: they are what a deployment of this image must load, and qualification receipts name them.

## How it differs from glm53-mia

[`glm53-mia`](../glm53-mia/) follows Mia's kit: 110 imported patches on TensorFold `f350cd3e`, taken as a whole. This lane starts from newer upstream TensorFold and takes only the changes we choose, each a reviewed commit we can send upstream or drop. Weights, drafter and build pins are the same. The [Dockerfile](Dockerfile) follows glm53-mia's step for step; without a kit its kit arguments are empty, the image's `tf.kit.*` labels are empty, and there are no kit notices to copy.

## Recommended serving profile

Releases of this lane are qualified with these settings. They live in home-lab's deployment profiles, not here; set them the same on every rank.

- `TF_GLM_CHUNK_SOLO=4096 TF_GLM_PROMPT_PAIR=solo`: a prompt step takes two 4,096-row solo chunks in one pass.
- `TF_GLM_SHARED_BUDGETS=0`: shared rounds size each stream's drafts from its last grant plus slack (see [ashhart/TensorFold#2720](https://github.com/ashhart/TensorFold/issues/2720)).
- `TF_GLM_HC_FUSE=1`: a prompt call's `hc_post` also writes the next site's `hc_pre` partials, with the same bytes.
- `TF_GLM_ABSORB_MMA=1 TF_GLM_EXL3_BF16_SLOTS=1`: opt-in arithmetic. Tensor-core absorb and expand GEMMs, and bf16 MoE slot rows for EXL3 prompt chunks. Both change rounding, so outputs drift slightly from the fp32 defaults.
- `TF_GLM_DENSE_BITS=4 TF_GLM_HEAD_BITS=8`: the dense linears packed to 4 bits and the output head to 8 at load. This is lossy against the bf16 weights.
- `--kv-cache fp8`.
- The DFlash2 drafter from `recipe.json` with `--drafter-bits 4`.

Qualification receipts live with each release, in its `qualification.receipt`. This lane has no release yet.
