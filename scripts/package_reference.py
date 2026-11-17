#!/usr/bin/env python3
"""Package reproduced gym baselines; never called by either evaluation entry point."""
import json
from pathlib import Path

path = Path('scoring/reference.json')
ref = json.loads(path.read_text())
scores = json.loads(Path('scoring/scores_test.json').read_text())['experiments']
for exp, record in ref['experiments'].items():
    assert not scores[exp]['failed_methods'], scores[exp]['failed_methods']
    assert set(record['results']) == set(scores[exp]['results'])
    for method, values in record['results'].items():
        typ = values['type']
        record['results'][method] = {'type': typ, **scores[exp]['results'][method]}
path.write_text(json.dumps(ref, indent=2) + '\n')
print('Packaged hidden-test reference results; paper numbers unchanged.')
