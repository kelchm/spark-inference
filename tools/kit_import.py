#!/usr/bin/env python3
"""Import a recipe kit's engine patch series into a TensorFold checkout as ordinary commits.

Each patch is applied with `git apply` in filename order, exactly as the kit's image build applies it. Mail-formatted
patches keep their author, date and message; plain diffs are attributed to --default-author with the patch's own
preamble as the message. Every commit records its source in an `Origin:` trailer. A final commit carries the kit's
licence, notice and credits verbatim. The result is checked against an independent apply of the whole series, and
every message's own From: header must name the author its commit carries. The series digest is the SHA-256 of the
patch files concatenated in filename order (as recipe.json's kit.patches_sha256 records it). Commit and missing author
dates are the kit revision's commit date, so the same kit, base and committer always give the same commit SHAs.
Needs Python 3.12+.
"""
import argparse
import email
import email.policy
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile

if sys.version_info < (3, 12):
    sys.exit('kit_import.py needs Python 3.12 or newer (tarfile extraction filters)')


def run(*args, cwd, env=None, stdin=None, capture=False):
    return subprocess.run(args, cwd=cwd, env=env, input=stdin, check=True,
                          stdout=subprocess.PIPE if capture else None, text=True).stdout


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


KIT_DATE = None  # the kit revision's commit date, set in main()


def commit(repo, message, author=None, date=None):
    env = dict(os.environ, GIT_COMMITTER_DATE=KIT_DATE, GIT_AUTHOR_DATE=date or KIT_DATE)
    if author:
        env['GIT_AUTHOR_NAME'], env['GIT_AUTHOR_EMAIL'] = author
    run('git', 'add', '-A', cwd=repo)
    run('git', 'commit', '-q', '--no-verify', '--cleanup=whitespace', '-F', '-', cwd=repo, env=env, stdin=message)


def apply(repo, patch):
    run('git', 'apply', '--check', str(patch), cwd=repo)
    run('git', 'apply', str(patch), cwd=repo)


def preamble(text):
    """Free text before the first diff, minus mail headers."""
    head = '' if text.startswith('diff --git ') else text.split('\ndiff --git ', 1)[0]
    return '\n'.join(line for line in head.splitlines() if not re.match(r'^(From|Subject|Date): ', line)).strip()


def header_author(message):
    """The decoded display name in one message's own From: header, as git mailinfo reads it."""
    sender = email.message_from_string(message, policy=email.policy.default)['From']
    return sender.addresses[0].display_name if sender else None


def import_patch(repo, patch, origin, default_author):
    """Commit one patch; returns (header author, committed author) for every message whose header names one."""
    text = patch.read_text()
    trailer = f'Origin: {origin} {patch.name} sha256:{sha(patch)}'
    if not text.startswith(('From ', 'From: ')):
        apply(repo, patch)
        commit(repo, f'kit {patch.stem}\n\n{preamble(text)}\n\n{trailer}\n'.replace('\n\n\n', '\n\n'), default_author)
        return []
    authors = []
    with tempfile.TemporaryDirectory() as work:
        if text.startswith('From: '):  # header-style mail without an mbox separator: one message
            count = 1
            (Path(work) / '0001').write_text(text)
        else:
            count = int(run('git', 'mailsplit', '-o' + work, str(patch), cwd=repo, capture=True).strip())
        for index in range(1, count + 1):
            msg, diff = Path(work) / 'msg', Path(work) / 'diff'
            mail = (Path(work) / f'{index:04d}').read_text()
            info = run('git', 'mailinfo', str(msg), str(diff), cwd=repo, stdin=mail, capture=True)
            fields = dict(line.split(': ', 1) for line in info.splitlines() if ': ' in line)
            apply(repo, diff)
            part = f' (part {index}/{count})' if count > 1 else ''
            message = f"{fields['Subject']}\n\n{msg.read_text().strip()}\n\n{trailer}{part}\n"
            commit(repo, message.replace('\n\n\n', '\n\n'), (fields['Author'], fields['Email']), fields.get('Date'))
            authors.append((header_author(mail), fields['Author']))
    return authors


def series_sha256(patches):
    digest = hashlib.sha256()
    for patch in patches:
        digest.update(patch.read_bytes())
    return digest.hexdigest()


def reference_tree(repo, base, patches):
    """The series applied to `base` in a scratch copy, as the image build does; returns the scratch path."""
    out = Path(tempfile.mkdtemp(prefix='kit-import-ref-'))
    with tempfile.TemporaryFile() as archive:
        subprocess.run(['git', '-C', str(repo), 'archive', base], stdout=archive, check=True)
        archive.seek(0)
        with tarfile.open(fileobj=archive) as t:
            t.extractall(out, filter='data')
    for patch in patches:
        subprocess.run(['git', 'apply', str(patch)], cwd=out, check=True, stderr=subprocess.DEVNULL)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True, help='TensorFold checkout, clean, at no branch it may move')
    parser.add_argument('--kit', type=Path, required=True, help='kit checkout at the revision to import')
    parser.add_argument('--kit-repo', required=True, help='owner/name of the kit on GitHub')
    parser.add_argument('--base', required=True, help='engine commit the kit pins')
    parser.add_argument('--branch', required=True)
    parser.add_argument('--notices', required=True, help='directory in the engine tree for the kit notices')
    parser.add_argument('--default-author', required=True, help='"Name <email>" for patches without a From: line')
    parser.add_argument('--series-sha256', help='refuse unless the patch series has this digest')
    args = parser.parse_args()
    repo, kit = args.repo.resolve(), args.kit.resolve()
    global KIT_DATE
    rev = run('git', 'rev-parse', 'HEAD', cwd=kit, capture=True).strip()
    KIT_DATE = run('git', 'log', '-1', '--format=%cI', cwd=kit, capture=True).strip()
    name, email_ = re.fullmatch(r'(.*?) <(.*)>', args.default_author).groups()
    patches = sorted((kit / 'patches/engine').glob('*.patch'))
    series = series_sha256(patches)
    if args.series_sha256 and series != args.series_sha256:
        raise SystemExit(f'patch series digest {series} is not the expected {args.series_sha256}')
    if run('git', 'status', '--porcelain', cwd=repo, capture=True).strip():
        raise SystemExit('engine checkout is not clean')
    run('git', 'switch', '-q', '-c', args.branch, args.base, cwd=repo)
    origin = f'{args.kit_repo}@{rev} patches/engine/'
    misattributed = []
    for patch in patches:
        for declared, made in import_patch(repo, patch, origin.rstrip('/'), (name, email_)):
            if declared and declared != made:
                misattributed.append((patch.name, declared, made))
    notices = repo / args.notices
    notices.mkdir(parents=True, exist_ok=False)
    for file in ('LICENSE', 'NOTICE.md', 'CREDITS.md'):
        if (kit / file).is_file():
            (notices / file).write_bytes((kit / file).read_bytes())
    commit(repo, f'packaging: carry the {args.kit_repo} licence, notice and credits\n\n'
                 f'The preceding commits apply {args.kit_repo} at {rev}\n'
                 f'(patches/engine, {len(patches)} patches) to TensorFold {args.base}.\n'
                 'The kit asks that its notice and credits accompany modified versions; these are its files verbatim.\n')
    if misattributed:
        raise SystemExit(f'authorship mismatch: {misattributed}')
    ref = reference_tree(repo, args.base, patches)
    got = Path(tempfile.mkdtemp(prefix='kit-import-got-'))
    with tempfile.TemporaryFile() as archive:
        subprocess.run(['git', '-C', str(repo), 'archive', 'HEAD~1'], stdout=archive, check=True)
        archive.seek(0)
        with tarfile.open(fileobj=archive) as t:
            t.extractall(got, filter='data')
    if subprocess.run(['diff', '-r', '-q', str(ref), str(got)]).returncode != 0:
        raise SystemExit(f'imported tree differs from the applied series: {ref} vs {got}')
    count = int(run('git', 'rev-list', '--count', f'{args.base}..HEAD', cwd=repo, capture=True))
    print(f'{args.branch}: {count} commits on {args.base[:12]}; {len(patches)} patches, series sha256 {series}; '
          'tree equals the applied series; authors preserved')


if __name__ == '__main__':
    main()
