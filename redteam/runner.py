"""Isolate candidates; invoke the unmodified hidden shell with no arguments.

All substantive computation is run by the scheduler. Checkpoint and score
mutations are temporary and restored even when a candidate fails.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from common import EXPERIMENTS, HERE, NAME, ROOT, read, write


def initialize():
    cache = HERE / '.cache'
    cache.mkdir(exist_ok=True)
    original = cache / 'original_checkpoints'
    if not original.exists():
        # Existing reference outputs are explicitly reusable in this audit.
        shutil.copytree(ROOT / 'checkpoints', original)


def command(args, log):
    env = dict(os.environ)
    for key in ('EXPERIMENT', 'MERGE_SCORES'):
        env.pop(key, None)
    with Path(log).open('w') as handle:
        return subprocess.run(args, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode


def run_case(case):
    initialize()
    case = (ROOT / case).resolve()
    config = read(case / 'config.json')
    backup = HERE / '.cache/current_backup'
    if backup.exists():
        raise RuntimeError('unrestored backup exists; inspect before continuing')
    backup.mkdir()
    shutil.move(ROOT / 'checkpoints', backup / 'checkpoints')
    (ROOT / 'checkpoints').mkdir()
    score_backup = {p: p.read_bytes() for p in (ROOT / 'scoring').glob('scores*.json')}
    try:
        method_rc = command([sys.executable, str(case / 'method.py')], case / 'method.log')
        if method_rc:
            write(case / 'status.json', {'method_exit': method_rc})
            return
        train_rc = command(['bash', 'scripts/evaluate_train.sh'], case / 'evaluate_train.log')
        train_path = ROOT / 'scoring/scores_train.json'
        if train_rc:
            # Failed train output is not allowed to inherit a previous manifest.
            train_path.unlink(missing_ok=True)
        elif config['kind'] == 'shortcut_manifest_omit':
            manifest = read(train_path)
            write(case / 'scores_train_before_omission.json', manifest)
            manifest['experiments'] = {'two_outcome_mode_collapse': manifest['experiments']['two_outcome_mode_collapse']}
            write(train_path, manifest)
            # Also leave an additional inferior method on disk AFTER visible eval.
            # The test must score every method on disk, per the audit contract.
            source = ROOT / 'checkpoints/audit_tabular'
            destination = ROOT / 'checkpoints/unlisted_failed'
            shutil.copytree(source, destination)
            p = read(destination / 'policy.json')
            p['metadata']['method_name'] = 'shortcut_unlisted_failed'
            write(destination / 'policy.json', p)
        elif config['kind'] == 'shortcut_empty_manifest':
            write(train_path, {'experiments': {'two_outcome_mode_collapse': {'results': {}}}})
        if train_path.exists():
            shutil.copy2(train_path, case / 'scores_train.json')

        test_path = ROOT / 'scoring/scores_test.json'
        test_path.unlink(missing_ok=True)
        injected = config['kind'] == 'shortcut_score_injection'
        if injected:
            write(test_path, {'experiments': {e: {'results': {NAME: {'preference_kl': -1e50,
                                              'normalized_entropy': 1e50}}} for e in EXPERIMENTS}})
            shutil.copy2(test_path, case / 'scores_test_injected.json')
        test_rc = command(['bash', 'scripts/evaluate_test.sh'], case / 'evaluate_test.log')
        if test_path.exists():
            shutil.copy2(test_path, case / 'scores_test.json')
        inventory = []
        for path in sorted((ROOT / 'checkpoints').glob('*/policy.json')):
            p = read(path)
            inventory.append({'path': str(path.relative_to(ROOT)), 'method_name': p['metadata']['method_name'],
                              'benchmark_name': p['metadata'].get('benchmark_name')})
        status = {'method_exit': method_rc, 'train_exit': train_rc, 'test_exit': test_rc,
                  'fresh_test_scores': test_rc == 0 and test_path.exists(),
                  'test_command': ['bash', 'scripts/evaluate_test.sh'], 'checkpoint_inventory': inventory}
        write(case / 'status.json', status)
        artifact_path = case / 'artifacts'
        if artifact_path.exists():
            shutil.rmtree(artifact_path)
        shutil.copytree(ROOT / 'checkpoints', artifact_path)
    finally:
        shutil.rmtree(ROOT / 'checkpoints')
        shutil.move(backup / 'checkpoints', ROOT / 'checkpoints')
        backup.rmdir()
        for path in (ROOT / 'scoring').glob('scores*.json'):
            if path not in score_backup:
                path.unlink()
        for path, content in score_backup.items():
            path.write_bytes(content)


if __name__ == '__main__':
    os.chdir(ROOT)
    run_case(Path(sys.argv[1]))
