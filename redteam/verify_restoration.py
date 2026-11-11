"""Lightweight file-integrity and run-completeness audit; performs no model runs."""
from pathlib import Path
import hashlib
import json
import os
import stat
import subprocess

ROOT = Path('/home/user')
HERE = ROOT / 'redteam'
BASE = (HERE / 'initial_commit.txt').read_text().strip()

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)

def blob_hash(content):
    return hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()

inherited = {'README.md', 'manifest.json', *(f'trajectory/turn_{i:04d}_raw.jsonl' for i in range(1, 6))}
protocol = {'action.yaml'}
entries = {}
for record in git('ls-tree', '-rz', BASE).split(b'\0'):
    if not record:
        continue
    info, name = record.split(b'\t', 1)
    mode, kind, object_id = info.decode().split()
    entries[os.fsdecode(name)] = (mode, kind, object_id)

mismatches = []
checked = 0
for name, (mode, kind, expected) in entries.items():
    if name in inherited | protocol or name.startswith('redteam/') or name == 'REDTEAM.md':
        continue
    path = ROOT / name
    if mode == '120000':
        payload = os.fsencode(os.readlink(path)) if path.is_symlink() else None
    else:
        payload = path.read_bytes() if path.is_file() else None
    checked += 1
    if payload is None or blob_hash(payload) != expected:
        mismatches.append(name)
    elif mode in ('100644', '100755') and bool(path.stat().st_mode & 0o111) != (mode == '100755'):
        mismatches.append(name + ' (executable mode)')

extra = []
for name in os.fsdecode(git('ls-files', '-z')).split('\0'):
    if name and name not in entries and not name.startswith('redteam/') and name != 'REDTEAM.md':
        extra.append(name)

original = HERE / '.cache/original_checkpoints'
def file_hashes(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file()}
checkpoints_match = original.exists() and file_hashes(ROOT / 'checkpoints') == file_hashes(original)

runs = {}
for name in json.loads((HERE / 'cases.json').read_text()):
    folder = HERE / name
    status = json.loads((folder / 'status.json').read_text())
    assert status['test_command'] == ['bash', 'scripts/evaluate_test.sh'], name
    assert all((folder / filename).exists() for filename in ('run.sh', 'method.py', 'config.json', 'evaluate_test.log')), name
    assert (folder / 'scores_test.json').exists() == status['fresh_test_scores'], name
    runs[name] = {'test_exit': status['test_exit'], 'fresh_test_scores': status['fresh_test_scores']}

result = {'initial_commit': BASE, 'checked_original_tracked_files': checked,
          'changed_original_files': mismatches, 'extra_tracked_files_outside_audit': extra,
          'checkpoints_identical_to_original_cache': checkpoints_match,
          'inherited_dirty_paths_preserved': sorted(inherited),
          'inherited_devices_still_character_devices': all(stat.S_ISCHR((ROOT / p).stat().st_mode) for p in ('README.md', 'manifest.json')),
          'protocol_exception': 'action.yaml is restored for this check, then set to done for final handoff',
          'no_unrestored_runner_backup': not (HERE / '.cache/current_backup').exists(),
          'runs': runs, 'fresh_score_runs': sum(r['fresh_test_scores'] for r in runs.values())}
(HERE / 'restoration.json').write_text(json.dumps(result, indent=2) + '\n')
assert not mismatches and not extra and checkpoints_match and result['no_unrestored_runner_backup'], result
print(f'Integrity passed: {checked} original tracked files; checkpoints restored; {len(runs)} runs, {result["fresh_score_runs"]} fresh score files.')
