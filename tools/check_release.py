#!/usr/bin/env python3
"""Check release files against the recipe they name and the image they pin.

A release file is recipes/<lane>/releases/<lane>-<YYYY.MM.DD>[.n].json, .n numbering a second release that day. It must name an existing recipe commit, repeat
that commit's recipe.json pins exactly, and pin an image digest whose labels record the same lane, recipe commit,
engine commit and kit. Run from a full clone; the registry is read anonymously, so the image must be public.
Provenance is checked separately with `gh attestation verify`.
"""
import json
from pathlib import Path
import re
import subprocess
import sys
import urllib.request

PINNED = ('model', 'engine', 'kit', 'build', 'weights', 'drafter')
IMAGE = re.compile(r'ghcr\.io/(?P<repo>[a-z0-9._/-]+)@(?P<digest>sha256:[0-9a-f]{64})')
CONFIG_TYPES = 'application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, ' \
    'application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json'


def registry(path, token, accept=CONFIG_TYPES):
    request = urllib.request.Request(f'https://ghcr.io/v2/{path}', headers={'Authorization': f'Bearer {token}',
                                                                              'Accept': accept})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def labels(repo, digest):
    """The image config labels for an arm64 image, following an index if the digest names one."""
    url = f'https://ghcr.io/token?scope=repository:{repo}:pull'
    with urllib.request.urlopen(url, timeout=60) as response:
        token = json.load(response)['token']
    manifest = registry(f'{repo}/manifests/{digest}', token)
    if 'manifests' in manifest:
        arm = [m for m in manifest['manifests'] if m.get('platform', {}).get('architecture') == 'arm64']
        if len(arm) != 1:
            raise ValueError(f'{digest} is an index without exactly one arm64 image')
        manifest = registry(f"{repo}/manifests/{arm[0]['digest']}", token)
    config = registry(f"{repo}/blobs/{manifest['config']['digest']}", token, accept='*/*')
    return config.get('config', {}).get('Labels') or {}


def check(path):
    problems = []
    release = json.loads(path.read_text())
    lane = path.parent.parent.name
    if path.name != f"{release.get('release')}.json" or not re.fullmatch(rf'{lane}-\d{{4}}\.\d{{2}}\.\d{{2}}(\.\d+)?',
                                                                          release.get('release', '')):
        problems.append(f'name must be {lane}-YYYY.MM.DD[.n] and match the file name')
    if release.get('lane') != lane:
        problems.append(f'lane must be {lane}')
    commit = release.get('recipe_commit', '')
    shown = subprocess.run(['git', 'show', f'{commit}:recipes/{lane}/recipe.json'], capture_output=True, text=True)
    if not re.fullmatch(r'[0-9a-f]{40}', commit) or shown.returncode:
        problems.append(f'recipe_commit {commit!r} has no recipes/{lane}/recipe.json')
    else:
        recipe = json.loads(shown.stdout)
        problems += [f'{key} differs from recipe.json at {commit[:12]}' for key in PINNED
                     if release.get(key) != recipe.get(key)]
    image = IMAGE.fullmatch(release.get('image', ''))
    if not image or image['repo'] != f'kelchm/spark-inference/{lane}':
        problems.append(f'image must be ghcr.io/kelchm/spark-inference/{lane}@sha256:<digest>')
    elif not problems:
        found = labels(image['repo'], image['digest'])
        want = {'recipe.lane': lane, 'org.opencontainers.image.revision': commit,
                'tf.repo': f"https://github.com/{release['engine']['repository']}.git",
                'tf.ref': release['engine']['commit'], 'tf.kit.repo': release['kit']['repository'],
                'tf.kit.rev': release['kit']['revision'],
                'tf.kit.series_sha256': release['kit']['patches_sha256']}
        problems += [f'image label {k} is {found.get(k)!r}, expected {v!r}' for k, v in want.items() if found.get(k) != v]
    qualification = release.get('qualification')
    if not isinstance(qualification, dict) or qualification.get('status') not in ('pending', 'qualified', 'rejected'):
        problems.append('qualification.status must be pending, qualified or rejected')
    return problems


def main():
    paths = [Path(p) for p in sys.argv[1:]] or sorted(Path('recipes').glob('*/releases/*.json'))
    failed = False
    for path in paths:
        problems = check(path)
        print(f"{path}: {'ok' if not problems else 'FAILED'}")
        for problem in problems:
            print(f'  - {problem}')
        failed |= bool(problems)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
