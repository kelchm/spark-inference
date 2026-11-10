#!/usr/bin/env python3
"""Check each lane's recipe against the engine fork it pins.

For recipes/<lane>/recipe.json: the engine commit must be on the lane's deploy branch. For a lane with a kit, the
kit-notices commit below it must name the recipe's kit revision, patch count and upstream engine commit, and every
patch it names must appear as imported commits carrying that kit's Origin trailer; our own changes are the commits
above the notices commit. A lane without a kit builds on upstream TensorFold itself: the upstream commit must be an
ancestor of the engine commit, and our own changes are the commits above it. Either way they must be exactly the
recipe's "carries" list, oldest first, each listing its commit's Adapted-from trailers as "adapted_from".

Every project whose patches a lane takes, its kit's or a source a carry adapts, must have its LICENSE, NOTICE.md and
CREDITS.md under packaging/kits/<repository name, lowercased>/ at the engine commit, as kit imports leave them; the
lane's image copies that folder. A recipe whose engine commit predates them lists those projects in "notices_pending",
exactly, and cannot be released. Only commit metadata and the trees of those folders are fetched.
"""
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

DEPTH = 600  # comfortably more than a kit's patch count plus what a lane carries
ADAPTED = re.compile(r'^Adapted-from: (.+)$', re.M)
NOTICES = ('LICENSE', 'NOTICE.md', 'CREDITS.md')


def git(*args, cwd):
    return subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def commits(repo, revs):
    """(sha, subject, body) from the tip of revs downward, as far as the shallow fetch reaches."""
    out = git('log', '--format=%H%x00%s%x00%B%x1e', revs, cwd=repo)
    return [tuple(entry.strip('\n').split('\x00', 2)) for entry in out.split('\x1e') if entry.strip()]


def compare_carries(recipe, ours, above):
    """Problems where our own commits (sha, subject, body), oldest first, differ from the recipe's carries."""
    subjects = [subject for _, subject, _ in ours]
    carries = recipe.get('carries', [])
    carried = [entry.get('subject') for entry in carries]
    if subjects != carried:
        return [f'commits above {above} are {subjects}; recipe carries {carried}']
    return [f"carry {subject!r}: adapted_from is {entry.get('adapted_from', [])}, its commit says {sources}"
            for (_, subject, body), entry in zip(ours, carries)
            if (sources := ADAPTED.findall(body)) != entry.get('adapted_from', [])]


def check_notices(recipe, repo):
    """Problems where the engine commit's kit notices differ from what the recipe takes and lists as pending."""
    kit = recipe.get('kit')
    sources = {kit['repository']} if kit else set()
    sources |= {source.partition('@')[0] for entry in recipe.get('carries', [])
                for source in entry.get('adapted_from', [])}
    missing = {}
    for source in sources:
        folder = f"packaging/kits/{source.rpartition('/')[2].lower()}"
        found = git('ls-tree', '--name-only', recipe['engine']['commit'], f'{folder}/', cwd=repo).split()
        if not {f'{folder}/{name}' for name in NOTICES} <= set(found):
            missing[source] = folder
    pending = set(recipe.get('notices_pending', []))
    return ([f"engine.commit lacks {missing[source]}/{{{','.join(NOTICES)}}}, the notices of {source}, whose patches "
             'this lane takes' for source in sorted(missing.keys() - pending)]
            + [f'notices_pending names {source}, but engine.commit carries its notices or the lane takes none of its '
               'patches: remove it' for source in sorted(pending - missing.keys())])


def check(path):
    recipe = json.loads(path.read_text())
    lane, engine, kit = path.parent.name, recipe['engine'], recipe.get('kit')
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
        problems += check_notices(recipe, repo)
        if kit is None:
            on_base = subprocess.run(['git', 'merge-base', '--is-ancestor', base, engine['commit']], cwd=repo,
                                     capture_output=True)
            if not re.fullmatch(r'[0-9a-f]{40}', base) or on_base.returncode:
                return problems + [f"engine.upstream {base[:12]!r} is not an ancestor of engine.commit within {DEPTH} "
                                   'commits']
            ours = list(reversed(commits(repo, f"{base}..{engine['commit']}")))
            return problems + compare_carries(recipe, ours, f'upstream {base[:12]}')
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
    return problems + compare_carries(recipe, list(reversed(log[:at])), 'the kit')


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
