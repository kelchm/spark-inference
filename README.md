# spark-inference

Recipes, image builds and qualification for the models I serve on a pair of NVIDIA DGX Sparks.

## Where things live

| Repository | Owns |
| --- | --- |
| [kelchm/TensorFold](https://github.com/kelchm/TensorFold) | Engine source. One `deploy/<model>` branch per model line: a recipe kit's patch series imported as commits on the engine commit it pins, plus backports of our own changes. `contrib/<feature>` branches carry work for upstream [ashhart/TensorFold](https://github.com/ashhart/TensorFold). |
| **spark-inference** (this repository) | Per-model recipes: the engine commit and kit each line builds, Dockerfiles, launch and readiness settings, qualification harnesses and their receipts, and every image build. |
| home-lab | Deployment only: which release of this repository runs where, host configuration, rollout and rollback. |

Engine behaviour and its tests belong in the engine fork. Model and hardware qualification belongs here. Nothing here patches engine source.

## Layout

- `models/<model>/recipe.json`: the pins for one model line (engine repository and commit, kit, base images, toolchain, weights).
- `models/<model>/Dockerfile`: builds that line's image from its pinned engine commit.
- `tools/`: maintenance tools, such as importing a kit's patch series into the engine fork.
- `.github/workflows/`: one reusable image build, called for each model line. Lint with `mise exec -- actionlint`; `mise.toml` pins the tools.

## Images

Each model line builds once, on GitHub's ARM64 runners. A pull request from this repository builds its head commit and publishes it to `ghcr.io/kelchm/spark-inference/<model>` as `sha-<commit>`, with a build provenance attestation; merging rebuilds nothing. Pull requests from forks build without credentials and publish nothing. Build layers are cached in the registry (`<model>:buildcache`), so a build that leaves the engine unchanged takes minutes. Deployments pin images by digest, never by tag. A successful build proves only that the image builds; qualification on the Sparks is a separate step with its own receipts.

## Licence

Apache-2.0. Recipes adapted from other projects keep their notices; see [NOTICE](NOTICE).
