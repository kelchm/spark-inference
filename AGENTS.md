# Working on spark-inference

This repository is the front door for models served on the DGX Spark pair. Engine source lives in [kelchm/TensorFold](https://github.com/kelchm/TensorFold); deployment lives in home-lab. Read the README's **Start here** and **Naming** sections first. Every routine change below has a tool and a CI check; use them rather than editing by hand.

## Rules

- **Published images come only from CI on `main`.** Local builds of a Dockerfile are for development and testing; never push them anywhere. `tools/dev_build.sh <engine commit> <lane>` makes one.
- **Never hand-edit generated or pinned values.** Release files come from `tools/new_release.py`, the README status table from `tools/status.py`, and a lane's engine branch from `tools/kit_import.py` (or upstream TensorFold, for a lane without a kit) plus reviewed commits.
- **Names follow the README's Naming table.** Lanes are `<model>-<recipe line>` with no versions in them. A release's date is when we promote it.
- **Engine commits stay upstreamable.** In kelchm/TensorFold, commit under the user's GitHub noreply address with no AI co-author lines (TensorFold's rule). In this repository, end commits with the model's `Co-Authored-By` trailer as usual.
- **Imported work keeps its authors.** Never rewrite someone else's commit; put fixes in a separate commit on top. When adapting someone's change, credit them (`Co-authored-by:` for people, an `Adapted-from:` or `Backport-of:` trailer for the source).
- **No deployment from here.** Nothing in this repository touches the Sparks. Home-lab pins a release file and owns rollout and rollback.
- **Upstream is the user's voice.** Show the exact text before opening an upstream issue or pull request, or claiming work through ATC.

## Update a lane's kit

1. Check out the new kit revision.
2. In a TensorFold checkout, import it into a new branch: `tools/kit_import.py --repo … --kit … --kit-repo … --base <commit the kit pins> --branch import/<lane>-<rev7> --notices packaging/kits/<kit> --default-author '<kit owner noreply>'`. It prints the patch count and series digest for the recipe. To reproduce an existing import, pass `--series-sha256` with the recipe's digest.
3. Re-apply the lane's carried commits (the recipe's `carries`, oldest first) on top, then compare with `git range-diff` against `deploy/<lane>`.
4. After review, move `deploy/<lane>` to the result (`git push --force-with-lease`). Every earlier release's engine commit stays reachable through its `release/<release>` tag.
5. In this repository, update the lane's `recipe.json`: `engine.commit`, `engine.upstream`, `kit.revision`, `kit.patches`, `kit.patches_sha256`, and `carries` if needed. Open a pull request. CI runs `tools/check_recipe.py` against the fork and builds the image.

## Carry an engine change

1. **Upstream first:** branch `contrib/<topic>` from upstream `main`, with a failing test before the fix, following TensorFold's AGENTS.md and pull request pre-flight.
2. **Backport:** branch `work/<lane>/<topic>` from `deploy/<lane>`, apply the change adapted to the lane's base with a `Backport-of: ashhart/TensorFold#<n>` trailer (or the contrib branch until the pull request exists), and open a pull request into `deploy/<lane>` in the fork.
3. After it merges, add it to the recipe's `carries` (`subject`; `upstream` once known; `adapted_from` repeating the commit's `Adapted-from:` trailers, which `tools/check_recipe.py` compares), bump `engine.commit`, and open a pull request here. A commit that adapts another project's Apache-2.0 patch needs that project's `LICENSE`, `NOTICE.md` and `CREDITS.md` under `packaging/kits/<repository name, lowercased>/` on the deploy branch, verbatim; the check requires them, and a recipe may list the project in `notices_pending` only until they land.
4. When upstream or the lane's kit includes the change, drop it at the next kit update (or, for a lane without a kit, the next rebase onto upstream) and remove it from `carries`.

## Cut a release

1. Merge the recipe change to `main` and wait for its publish run.
2. `tools/new_release.py <lane>` writes `recipes/<lane>/releases/<lane>-<date>.json` and refreshes the README status. Check the `changes` line says what a reader needs.
3. Create and push the `release/<release>` tag on the engine commit in the fork, as the tool prints.
4. Open a pull request. CI checks the release against its recipe commit, the image's labels and provenance, and the fork tag.

## Record qualification

After home-lab qualifies or rejects a release on the Sparks, set its `qualification` to `{"status": "qualified" | "rejected", "receipt": "<link>", "date": "<YYYY-MM-DD>"}`, run `tools/status.py`, and open a pull request.

## Checks

`mise exec -- actionlint` for workflows. `python3 tools/check_recipe.py`, `python3 tools/check_release.py` and `python3 tools/status.py --check` are what CI runs.
