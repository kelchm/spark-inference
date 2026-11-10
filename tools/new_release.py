#!/usr/bin/env python3
"""Create a lane's next release file from an image that main has already published.

The release pins the image that CI published for the recipe commit (tag sha-<commit>, resolved to its digest),
repeats that commit's recipe pins, is named <lane>-<today's UTC date>[.n], and starts with qualification pending.
It prints the release/<name> tag to create on the engine commit in the fork, which the release check requires.
The date is when we promote, not when a kit or engine was published. Then regenerate the README status and open a
pull request; tools/check_release.py and CI check the result.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

PINNED = ('model', 'engine', 'kit', 'carries', 'build', 'weights', 'drafter')
INDEX_TYPES = 'application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json'


def git(*args):
    return subprocess.run(['git', *args], check=True, capture_output=True, text=True).stdout.strip()


def published_digest(lane, commit):
    repo = f'kelchm/spark-inference/{lane}'
    with urllib.request.urlopen(f'https://ghcr.io/token?scope=repository:{repo}:pull', timeout=60) as response:
        token = json.load(response)['token']
    request = urllib.request.Request(f'https://ghcr.io/v2/{repo}/manifests/sha-{commit}', method='HEAD',
                                     headers={'Authorization': f'Bearer {token}', 'Accept': INDEX_TYPES})
    with urllib.request.urlopen(request, timeout=60) as response:
        return f"ghcr.io/{repo}@{response.headers['Docker-Content-Digest']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lane')
    parser.add_argument('--commit', help='recipe commit on main that published the image (default: the latest one '
                        "that changed the lane's recipe)")
    parser.add_argument('--changes', help='what changed since the previous release (default: those commit subjects)')
    args = parser.parse_args()
    lane, recipe_path = args.lane, f'recipes/{args.lane}'
    git('fetch', '-q', 'origin', 'main')
    commit = args.commit or git('log', '-1', '--format=%H', 'origin/main', '--', recipe_path,
                                f':(exclude){recipe_path}/releases')
    recipe = json.loads(git('show', f'{commit}:{recipe_path}/recipe.json'))
    if recipe.get('notices_pending'):
        sys.exit(f"{commit[:12]}'s image lacks the notices of {', '.join(recipe['notices_pending'])} (notices_pending): "
                 'move the engine commit to one that carries them first')
    releases = Path(recipe_path, 'releases')
    previous = sorted(releases.glob('*.json'), key=lambda p: [int(x) for x in p.stem.removeprefix(f'{lane}-').split('.')])
    changes = args.changes
    if not changes:
        if previous:
            since = json.loads(previous[-1].read_text())['recipe_commit']
            changes = '; '.join(git('log', '--reverse', '--format=%s', f'{since}..{commit}', '--', recipe_path,
                                    f':(exclude){recipe_path}/releases').splitlines()) or 'No recipe changes; republished.'
        else:
            changes = 'First release of this lane.'
    name = f"{lane}-{datetime.now(timezone.utc):%Y.%m.%d}"
    n = 1
    while (releases / f'{name}{"" if n == 1 else f".{n}"}.json').exists():
        n += 1
    name = name if n == 1 else f'{name}.{n}'
    release = {'release': name, 'lane': lane, 'image': published_digest(lane, commit), 'recipe_commit': commit,
               **{k: recipe[k] for k in PINNED if k in recipe}, 'changes': changes,
               'qualification': {'status': 'pending'}}
    releases.mkdir(parents=True, exist_ok=True)
    (releases / f'{name}.json').write_text(json.dumps(release, indent=2) + '\n')
    subprocess.run(['python3', 'tools/status.py'], check=True)
    print(f'Wrote {releases / name}.json pinning {release["image"]}')
    engine = release['engine']
    print('Tag its engine commit in the fork, then open the pull request (CI checks the tag):')
    print(f"  git -C <TensorFold checkout> tag -a release/{name} {engine['commit']} -m 'Engine commit of release {name}'")
    print(f"  git -C <TensorFold checkout> push <fork remote> refs/tags/release/{name}")


if __name__ == '__main__':
    main()
