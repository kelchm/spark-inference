#!/usr/bin/env python3
"""Render a release-pinned SparkRun profile and Mia's thinking-only effort template."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from string import Template

ROOT = Path(__file__).resolve().parents[1]
BASE_LINE = "{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else 'max' -%}"
EFFORT_LINE = "{%- set effective_reasoning_effort = (reasoning_effort if reasoning_effort in ['low', 'high'] else 'max') if (reasoning_effort is defined and reasoning_effort is not none) else none -%}"


def outputs(release_path):
    release_path = Path(release_path).resolve()
    release = json.loads(release_path.read_text())
    lane, name = release['lane'], release['release']
    if lane != 'glm53-mia' or not re.fullmatch(r'glm53-mia-\d{4}\.\d{2}\.\d{2}(?:\.\d+)?', name):
        raise ValueError('Unsupported release lane or name')
    if not re.fullmatch(r'ghcr\.io/kelchm/spark-inference/glm53-mia@sha256:[0-9a-f]{64}', release['image']):
        raise ValueError('The deployment requires a digest-pinned lane image')
    for group in ('weights', 'drafter'):
        if not re.fullmatch(r'[\w.-]+/[\w.-]+', release[group]['repository']) or not re.fullmatch(r'[0-9a-f]{40}', release[group]['revision']):
            raise ValueError('Invalid immutable model input')
    base = ROOT / 'recipes' / lane / 'sparkrun'
    mod = base / 'mods/glm53-native'
    source = (mod / 'upstream_chat_template.jinja').read_text()
    if source.count(BASE_LINE) != 1:
        raise ValueError('Mia effort-line input differs; refusing an unrecognized template')
    rendered = source.replace(BASE_LINE, EFFORT_LINE)
    profile = name + '-agent-1m'
    values = {k: json.dumps(v) for k, v in {
        'model': release['weights']['repository'], 'model_revision': release['weights']['revision'],
        'image': release['image'], 'profile': profile, 'release': name,
        'drafter': release['drafter']['repository'], 'drafter_revision': release['drafter']['revision'],
        'engine_commit': release['engine']['commit'], 'kit_revision': release['kit']['revision'],
        'learn_dir': '/cache/huggingface/spark-inference/learn/' + profile,
    }.items()}
    profiles = {}
    for serving, quota in (('agent-1m', 64), ('quota-probe', 1)):
        preset = dict(values, profile=json.dumps(name + '-' + serving), serving_profile=json.dumps(serving),
                      learn_gib=json.dumps(str(quota)),
                      learn_dir=json.dumps('/cache/huggingface/spark-inference/learn/' + name + '-' + serving))
        profiles[base / 'profiles' / (name + '-' + serving + '.yaml')] = Template((base / 'profile.yaml.in').read_text()).substitute(preset)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(mod.iterdir())
             if p.is_file() and p.name not in ('manifest.json', 'chat_template.jinja')}
    files['chat_template.jinja'] = hashlib.sha256(rendered.encode()).hexdigest()
    manifest = {'schema': 'spark-inference-runtime/v1', 'lane': lane,
                'template_origin': {'repository': release['kit']['repository'], 'revision': release['kit']['revision'],
                                    'path': 'templates/glm-5.3-flash-vision.jinja',
                                    'sha256': files['upstream_chat_template.jinja'], 'effort_line': 'thinking'},
                'files': dict(sorted(files.items()))}
    return profiles | {mod / 'chat_template.jinja': rendered,
                       mod / 'manifest.json': json.dumps(manifest, indent=2) + '\n'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('release', type=Path)
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    for path, text in outputs(args.release).items():
        if args.check:
            if not path.is_file() or path.read_text() != text:
                raise SystemExit('Stale generated input: ' + str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        print(path.relative_to(ROOT))


if __name__ == '__main__':
    main()
