# GLM-5.3-Flash on two DGX Sparks

The engine is [`deploy/glm53`](https://github.com/kelchm/TensorFold/tree/deploy/glm53) in kelchm/TensorFold: TensorFold `f350cd3e` with [MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104](https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104) imported as commits (110 patches at `13f2464`, authors preserved, kit notices under `packaging/kits/`). It carries no changes of our own yet. [recipe.json](recipe.json) pins the exact engine commit, the kit, base images, toolchain and weights.

## Intended settings

`KV_SHARD=1`, `CACHE_FIRST=1` with a small prompt cache, `--learn-conversations` on, thinking kept (`CLEAR_THINKING=0`), video off. Mia's README measures the KV-sharded pool at about 3.15M tokens on her pair; ours is whatever the startup line reports.

## Not yet qualified

A built image proves only that it builds. Before this line serves production:

- Conversations containing images must persist to disk and restore correctly. Upstream refuses image states on disk today, and this kit's base predates that refusal; this needs engine work first.
- On the Sparks: capacity at four streams and a 1M window; a single large first save and growth toward 1M tokens; resume after restart and after eviction; a process killed mid-save; the disk cap on both Sparks; and prompt-cache hits across Pi tool loops.

## Updating the kit

Import a new kit revision into a new branch with [tools/kit_import.py](../../tools/kit_import.py), compare it with the current branch using `git range-diff`, then point `recipe.json` at the result.
