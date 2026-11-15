#!/usr/bin/env bash
# Development image of one lane from a pushed engine commit, built on this workstation with the lane's Dockerfile and
# recipe pins (base images, toolchain, kit identity). Loaded into the local Docker only, never pushed to a registry:
# published images come only from CI on main (see AGENTS.md). The image is labelled with the lane and a dev- recipe
# revision, so it cannot pass for a published build.
#
#   tools/dev_build.sh <engine-commit> <lane>
#
# Environment: DOCKER_CONTEXT (default colima-tf-build), ZIG_JOBS (default: all of the builder's CPUs).
set -euo pipefail
usage="usage: tools/dev_build.sh <engine-commit> <lane>"
sha=${1:?${usage}}
lane=${2:?${usage}}
root=$(cd "$(dirname "$0")/.." && pwd)
recipe="${root}/recipes/${lane}/recipe.json"
ctx=${DOCKER_CONTEXT:-colima-tf-build}
[[ "${sha}" =~ ^[0-9a-f]{40}$ ]] || { echo "engine commit must be a full 40-character SHA" >&2; exit 1; }
[[ -f "${recipe}" ]] || { echo "no recipe for lane ${lane}: ${recipe}" >&2; exit 1; }
pin() { jq -er "$1" "${recipe}"; }
kit() { jq -r ".kit.$1 // \"\"" "${recipe}"; }
repo="https://github.com/$(pin .engine.repository).git"
grep -q "^${sha}" <<< "$(git ls-remote "${repo}")" \
  || echo "note: ${sha} is not a branch or tag tip on ${repo}; the build fetches it by SHA" >&2
revision="dev-$(git -C "${root}" rev-parse --short HEAD)"
[[ -z "$(git -C "${root}" status --porcelain)" ]] || revision+="-dirty"
tag="tensorfold-dev:${lane}-${sha:0:12}"
start=${SECONDS}
docker --context "${ctx}" buildx build --load \
  --file "${root}/recipes/${lane}/Dockerfile" \
  --build-arg BASE_IMAGE="$(pin .build.base_image)" \
  --build-arg RUNTIME_IMAGE="$(pin .build.runtime_image)" \
  --build-arg ZIG_VERSION="$(pin .build.zig_version)" \
  --build-arg ZIG_SHA256="$(pin .build.zig_sha256)" \
  --build-arg ZIG_CPU="$(pin .build.zig_cpu)" \
  --build-arg SM="$(pin .build.sm)" \
  ${ZIG_JOBS:+--build-arg ZIG_JOBS="${ZIG_JOBS}"} \
  --build-arg TF_REPO="${repo}" \
  --build-arg TF_REF="${sha}" \
  --build-arg RECIPE_COMMIT="${revision}" \
  --build-arg RECIPE_LANE="${lane}" \
  --build-arg KIT_REPO="$(kit repository)" \
  --build-arg KIT_REV="$(kit revision)" \
  --build-arg KIT_SERIES_SHA256="$(kit patches_sha256)" \
  --tag "${tag}" \
  "${root}/recipes/${lane}"
echo "built ${tag} (${lane}) in $((SECONDS - start)) s: $(docker --context "${ctx}" image inspect --format '{{.Id}}' "${tag}")"
