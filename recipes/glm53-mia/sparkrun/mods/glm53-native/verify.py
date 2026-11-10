#!/usr/bin/env python3
"""Refuse missing, changed or linked runtime-mod inputs before installation."""
import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root)
    manifest = root / 'manifest.json'
    if manifest.is_symlink():
        raise ValueError('Linked runtime manifest')
    expected = json.loads(manifest.read_text())['files']
    actual = {p.name for p in root.iterdir() if p.is_file() and p.name != 'manifest.json'}
    if actual != set(expected):
        raise ValueError('Runtime mod file inventory differs')
    for name, digest in expected.items():
        p = root / name
        if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed runtime mod input: ' + name)
    return expected


if __name__ == '__main__':
    verify(Path(__file__).resolve().parent)
