from pathlib import Path
import json
import subprocess
import time

root = Path('/home/user')
cases = json.loads((root / 'redteam/cases.json').read_text())
summary = []
for name in cases:
    print(f'RUN {name}', flush=True)
    start = time.time()
    result = subprocess.run(['bash', f'redteam/{name}/run.sh'], cwd=root)
    entry = {'case': name, 'runner_exit': result.returncode, 'seconds': time.time() - start}
    summary.append(entry)
    (root / 'redteam/batch_status.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(entry, flush=True)
print('All cases completed', flush=True)
