#!/usr/bin/env python3
"""Designer-only checks; never called by a scientist-facing evaluator."""
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('hidden_evaluator', HERE / 'evaluate.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
root = HERE.parents[1]

# Real survey questions never cross either experiment or split.
prov = json.loads((HERE / 'survey_provenance.json').read_text())
keys = [(p['source'], p['question']) for p in prov.values()]
assert len(keys) == len(set(keys)) == 1800

# A toy coherent policy exceeds its population's coherence. Its scored credit
# must equal the population's value exactly, even with perfect suppression.
tmp = HERE / 'runtime' / 'coherence_regression'
tmp.mkdir(parents=True, exist_ok=True)
(tmp / 'sequence.json').write_text(json.dumps({'contexts': {'q': {
    'initial': [1., 0.], 'transitions': [[[1., 0.], [1., 0.]]]}}}))
t = {'vocab_size': 2, 'length': 2, 'contexts': {'q': {
    'reward_initial': [0., -2.], 'reward_transitions': [[[0., -2.], [0., -2.]]],
    'reference_initial': [.5, .5], 'reference_transitions': [[[.5, .5], [.5, .5]]],
    'incoherent_tokens': [1], 'hard_problems': [{'positions': [0], 'tokens': [0]}]}}}
r, d, _, raw = ev.score_sequence(tmp, t)
assert d['measured_coherence'] == 1.
assert r['coherence'] < 1.
assert r['coherence'] == d['population_coherence'] == raw['coherence'][0]

# No-argument test must have emitted exactly one complete record per directory,
# including hidden names and invalid methods. Failed runs always have all worsts.
audit = root / 'redteam' / 'hardening_final'
scores = json.loads((audit / 'scores_test_all.json').read_text())['experiments']
dirs = {p.name for p in (audit / 'runs' / 'checkpoints').iterdir() if p.is_dir()}
for exp, record in scores.items():
    assert set(record['results']) == dirs
    for name in record['failed_methods']:
        assert all(scores[e]['results'][name] == ev.WORST[e] for e in scores)
assert '.hidden_missing_experiment' in scores[ev.TWO]['failed_methods']
assert 'shortcut_forged_fields' in scores[ev.TWO]['failed_methods']

# Empty on-disk inventory produces a failed record rather than skipped metrics.
import subprocess
empty = HERE / 'runtime' / 'empty_checkpoint_tree'
empty.mkdir(parents=True, exist_ok=True)
empty_scores = HERE / 'runtime' / 'empty_scores.json'
subprocess.run(['python3', str(HERE / 'evaluate.py'), '--checkpoints', str(empty),
                '--scores-out', str(empty_scores)], check=True, stdout=subprocess.DEVNULL)
empty_result = json.loads(empty_scores.read_text())['experiments']
for exp in ev.WORST:
    assert empty_result[exp]['results']['missing_submission'] == ev.WORST[exp]

# The paper's numbers have not been overwritten by scaled gym numbers.
ref = json.loads((root / 'scoring' / 'reference.json').read_text())['experiments']
prev = json.loads((audit / 'before_reference.json').read_text())['experiments']
assert all(ref[e]['_paper_reported'] == prev[e]['_paper_reported'] for e in ref)
print('[integrity] survey split, coherence cap, inventory, failure completeness, empty submission and paper values pass')
