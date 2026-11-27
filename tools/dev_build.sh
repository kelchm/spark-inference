#!/usr/bin/env bash
# Development image of one lane from a pushed engine commit, built on this workstation with the lane's Dockerfile and
# recipe pins (base images, toolchain, kit identity). Loaded into the local Docker only, never pushed to a registry:
# published images come only from CI on main (see AGENTS.md). The image is labelled with the lane and a dev- recipe
# revision, so it cannot pass for a published build.
#
#   tools/dev_build.sh <engine-commit> <lane>
#
# Environment:
#   DOCKER_CONTEXT  Docker buildx context (default: colima-tf-build)
#   ZIG_JOBS        Parallel Zig build jobs (default: all builder CPUs)
#   SM              CUDA SM target override (default: recipe .build.sm, e.g. 121 for GB10)
#   TOOLS           1 to build CUDA dev tools (default: 1); 0 for faster server-only dev builds
#   FATBINS_DIR     Optional directory of prebuilt <name>.fatbin files (-Dfatbins fast rebuild path)
#   TP_FATBIN       Optional path to prebuilt tp_mailbox.fatbin (-Dtp-fatbin)
#   TP_NVCC         1 to compile the TP mailbox fatbin with nvcc when TP_FATBIN is unset (default: 0)
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
sm="${SM:-$(pin .build.sm)}"
tools="${TOOLS:-1}"
tp_nvcc="${TP_NVCC:-0}"

extra_flags=()
staging_dir=""
cleanup() {
  if [[ -n "${staging_dir}" && -d "${staging_dir}" ]]; then
    rm -rf "${staging_dir}"
  fi
}
trap cleanup EXIT

if [[ -n "${FATBINS_DIR:-}" || -n "${TP_FATBIN:-}" ]]; then
  fatbins_ctx=""
  if [[ -n "${FATBINS_DIR:-}" ]]; then
    [[ -d "${FATBINS_DIR}" ]] || { echo "FATBINS_DIR does not exist: ${FATBINS_DIR}" >&2; exit 1; }
    extra_flags+=(--build-arg USE_PREBUILT_FATBINS=1)
    fatbins_ctx="${FATBINS_DIR}"
  fi
  if [[ -n "${TP_FATBIN:-}" ]]; then
    [[ -f "${TP_FATBIN}" ]] || { echo "TP_FATBIN does not exist: ${TP_FATBIN}" >&2; exit 1; }
    extra_flags+=(--build-arg USE_PREBUILT_TP_FATBIN=1)
    if [[ -n "${FATBINS_DIR:-}" && "${TP_FATBIN}" == "${FATBINS_DIR%/}/tp_mailbox.fatbin" ]]; then
      fatbins_ctx="${FATBINS_DIR}"
    else
      staging_dir=$(mktemp -d)
      if [[ -n "${FATBINS_DIR:-}" ]]; then
        shopt -s nullglob
        for f in "${FATBINS_DIR}"/*.fatbin; do
          cp -L "${f}" "${staging_dir}/"
        done
      fi
      cp -L "${TP_FATBIN}" "${staging_dir}/tp_mailbox.fatbin"
      fatbins_ctx="${staging_dir}"
    fi
  fi
  extra_flags+=(--build-context "prebuilt-fatbins=${fatbins_ctx}")
fi

if [[ "${tp_nvcc}" == 1 ]]; then
  extra_flags+=(--build-arg TP_NVCC=1)
fi

start=${SECONDS}
docker --context "${ctx}" buildx build --load \
  --file "${root}/recipes/${lane}/Dockerfile" \
  --build-arg BASE_IMAGE="$(pin .build.base_image)" \
  --build-arg RUNTIME_IMAGE="$(pin .build.runtime_image)" \
  --build-arg ZIG_VERSION="$(pin .build.zig_version)" \
  --build-arg ZIG_SHA256="$(pin .build.zig_sha256)" \
  --build-arg ZIG_CPU="$(pin .build.zig_cpu)" \
  --build-arg SM="${sm}" \
  --build-arg TOOLS="${tools}" \
  ${ZIG_JOBS:+--build-arg ZIG_JOBS="${ZIG_JOBS}"} \
  --build-arg TF_REPO="${repo}" \
  --build-arg TF_REF="${sha}" \
  --build-arg RECIPE_COMMIT="${revision}" \
  --build-arg RECIPE_LANE="${lane}" \
  --build-arg KIT_REPO="$(kit repository)" \
  --build-arg KIT_REV="$(kit revision)" \
  --build-arg KIT_SERIES_SHA256="$(kit patches_sha256)" \
  ${extra_flags[@]+"${extra_flags[@]}"} \
  --tag "${tag}" \
  "${root}"
echo "built ${tag} (${lane}) in $((SECONDS - start)) s: $(docker --context "${ctx}" image inspect --format '{{.Id}}' "${tag}")"
