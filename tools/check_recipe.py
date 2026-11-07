#!/usr/bin/env python3
"""Check each lane's recipe against the engine fork it pins.

For recipes/<lane>/recipe.json: the engine commit must be on the lane's deploy branch; the kit-notices commit below
it must name the recipe's kit revision, patch count and upstream engine commit; every patch it names must appear as
imported commits carrying that kit's Origin trailer; and the commits above the notices commit (our own changes) must
be exactly the recipe's "carries" list, oldest first. Only commit metadata is fetched.
"""
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

DEPTH = 600  # comfortably more than a kit's patch count plus what a lane carries


def git(*args, cwd):
    return subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def commits(repo, tip):
    """(sha, subject, body) from tip downward, as far as the shallow fetch reaches."""
    out = git('log', '--format=%H%x00%s%x00%B%x1e', tip, cwd=repo)
    return [tuple(entry.strip('\n').split('\x00', 2)) for entry in out.split('\x1e') if entry.strip()]


def check(path):
    recipe = json.loads(path.read_text())
    lane, engine, kit = path.parent.name, recipe['engine'], recipe['kit']
    problems = []
    if recipe.get('lane') != lane:
        problems.append(f'lane must be {lane}')
    if engine.get('branch') != f'deploy/{lane}':
        problems.append(f'engine.branch must be deploy/{lane}')
    base = engine.get('upstream', '').rpartition('@')[2]
    with tempfile.TemporaryDirectory() as repo:
        git('init', '-q', cwd=repo)
        url = f"https://github.com/{engine['repository']}.git"
        git('fetch', '-q', '--filter=tree:0', f'--depth={DEPTH}', url, f"+refs/heads/{engine['branch']}:refs/lane", cwd=repo)
        on_branch = subprocess.run(['git', 'merge-base', '--is-ancestor', engine['commit'], 'refs/lane'], cwd=repo)
        if on_branch.returncode:
            return problems + [f"engine.commit {engine['commit'][:12]} is not on {engine['branch']}"]
        log = commits(repo, engine['commit'])
    notice = f"packaging: carry the {kit['repository']} licence, notice and credits"
    at = next((i for i, (_, subject, _) in enumerate(log) if subject == notice), None)
    if at is None:
        return problems + [f'no "{notice}" commit within {DEPTH} commits of engine.commit']
    body = log[at][2]
    expected = (f"apply {kit['repository']} at {kit['revision']}", f"({kit.get('patches_dir', 'patches/engine')}, "
                f"{kit['patches']} patches) to TensorFold {base}")
    problems += [f'kit notices commit does not say "{text}"' for text in expected if text not in ' '.join(body.split())]
    origin = re.compile(rf"^Origin: {re.escape(kit['repository'])}@{kit['revision']} \S+ (\S+\.patch) sha256:", re.M)
    imported = {m.group(1) for _, _, b in log[at + 1:] for m in origin.finditer(b)}
    if len(imported) != kit['patches']:
        problems.append(f"{len(imported)} imported kit patches carry this kit's Origin trailer; recipe says {kit['patches']}")
    ours = [subject for _, subject, _ in reversed(log[:at])]
    carried = [entry.get('subject') for entry in recipe.get('carries', [])]
    if ours != carried:
        problems.append(f'commits above the kit are {ours}; recipe carries {carried}')
    return problems


def main():
    paths = [Path(p) for p in sys.argv[1:]] or sorted(Path('recipes').glob('*/recipe.json'))
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
