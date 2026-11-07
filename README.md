# spark-inference

Recipes, image builds and qualification for the models I serve on a pair of NVIDIA DGX Sparks.

## Where things live

| Repository | Owns |
| --- | --- |
| [kelchm/TensorFold](https://github.com/kelchm/TensorFold) | Engine source. One `deploy/<lane>` branch per lane: its base (a recipe kit imported as commits on the engine commit the kit pins, or upstream TensorFold itself) plus our own changes. `contrib/<topic>` branches carry work for upstream [ashhart/TensorFold](https://github.com/ashhart/TensorFold). |
| **spark-inference** (this repository) | Per-lane recipes: the engine commit and kit each lane builds, Dockerfiles, launch and readiness settings, qualification harnesses and their receipts, and every image build. |
| home-lab | Deployment only: which release of this repository runs where, host configuration, rollout and rollback. |

Engine behaviour and its tests belong in the engine fork. Model and hardware qualification belongs here. Nothing here patches engine source.

## Naming

A **lane** is one maintained way of serving a model: `<model>-<recipe line>`, where the recipe line names whose recipe it follows. `glm53-mia` follows Mia's GLM-5.3-Flash Spark kit; a recipe of our own on upstream TensorFold would be `glm53-kelchm`. Lane names carry no versions: kit dates, engine commits and TensorFold versions are contents, recorded in the recipe, the image labels and each release.

| Thing | Pattern | Example |
| --- | --- | --- |
| Engine branch | `deploy/<lane>` | `deploy/glm53-mia` |
| Engine experiment | `work/<lane>/<topic>` | `work/glm53-mia/prompt-accounting` |
| Upstream contribution | `contrib/<topic>`, based on upstream `main` | `contrib/prompt-accounting` |
| Recipe | `recipes/<lane>/` | `recipes/glm53-mia/` |
| Image | `ghcr.io/kelchm/spark-inference/<lane>` | `ghcr.io/kelchm/spark-inference/glm53-mia` |
| Image tag | `sha-<commit>` from `main` | `glm53-mia:sha-…` |
| Release | `<lane>-<YYYY.MM.DD>`, with `.2`, `.3`… for further releases that day; a file in `recipes/<lane>/releases/` | `glm53-mia-2026.11.07` |
| Deployment profile (home-lab) | `<release>-<serving profile>` | `glm53-mia-2026.11.07-agent-1m` |

Deployments pin a release, and a release pins an image digest; tags are only for finding builds.

## Layout

- `recipes/<lane>/recipe.json`: the pins for one lane (engine repository and commit, kit, base images, toolchain, weights).
- `recipes/<lane>/Dockerfile`: builds that lane's image from its pinned engine commit.
- `recipes/<lane>/releases/<release>.json`: a promoted build. It pins one published image by digest, repeats the recipe pins of the commit that built it, says what changed, and records qualification (`pending`, `qualified` or `rejected`). Deployments pin a release file. `tools/check_release.py` and the Releases workflow check every release against its recipe commit, the image's labels and its build provenance; release files never trigger a build.
- `tools/`: maintenance tools, such as importing a kit's patch series into the engine fork.
- `.github/workflows/`: one reusable image build, called for each lane. Lint with `mise exec -- actionlint`; `mise.toml` pins the tools.

## Images

Published images come only from CI on `main`, after review: each merge that changes a lane builds it on GitHub's ARM64 runners and publishes `ghcr.io/kelchm/spark-inference/<lane>` as `sha-<commit>`, with a build provenance attestation. Pull requests build to check and publish nothing; those from this repository write the registry build cache (`<lane>:buildcache`), so the build after merge reuses the compiled engine and takes minutes. Fork pull requests get no credentials. Deployments pin images by digest, never by tag. A successful build proves only that the image builds; qualification on the Sparks is a separate step with its own receipts. Local builds of the same Dockerfile are fine for development and testing, never for publishing.

## Licence

Apache-2.0. Recipes adapted from other projects keep their notices; see [NOTICE](NOTICE).
