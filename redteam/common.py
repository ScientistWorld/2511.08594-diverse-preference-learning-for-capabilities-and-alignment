"""Actual candidate implementations. Inputs are scientist-visible files only.

The audit runner, separately, invokes the restored test shell without arguments.
No implementation imports hidden evaluation, paper text, or reference scores.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path('/home/user')
HERE = ROOT / 'redteam'
EXPERIMENTS = {
    'two_outcome_mode_collapse': ('two_outcome_split_benchmark.json', 'two_outcome'),
    'tabular_preference_learning': ('multi_outcome_benchmark.json', 'tabular'),
    'calibration_multiple_choice': ('calibration_benchmark.json', 'calibration'),
}
NAME = 'audit_candidate'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n')


def sample(contexts, seed=11, enumerate_actions=False):
    rng = random.Random(seed)
    rows = []
    for context in contexts:
        actions = context['actions']
        choices = list(range(len(actions)))
        indices = ([i % len(actions) for i in range(128)] if enumerate_actions else
                   rng.choices(choices, [a['policy_probability'] for a in actions], k=128))
        for idx, selected in enumerate(indices):
            action = actions[selected]
            rows.append({'context_id': context['id'], 'sample_index': idx,
                         'action_id': action['id'],
                         'is_best_of_n_solution': action['is_best_of_n_solution']})
    return {'metadata': {'seed': seed, 'samples_per_context': 128}, 'samples': rows}


def train(benchmark, output, alpha=1., beta=.1, steps=6000, seed=11):
    subprocess.run([sys.executable, 'method/train_tabular_spl.py',
                    '--benchmark', str(benchmark), '--output-dir', str(output),
                    '--method-name', NAME, '--alpha', str(alpha), '--beta', str(beta),
                    '--steps', str(steps), '--lr', '.8', '--seed', str(seed)], check=True)


def produce(case_dir):
    case_dir = Path(case_dir)
    config = read(case_dir / 'config.json')
    kind = config['kind']
    for experiment, (filename, suffix) in EXPERIMENTS.items():
        output = ROOT / 'checkpoints' / ('audit_' + suffix)
        benchmark_path = ROOT / 'data' / filename
        benchmark = read(benchmark_path)
        if kind == 'train':
            train(benchmark_path, output, **config.get('training', {}))
            continue
        if kind == 'train_selected':
            # Fit from comparisons only. Choose hyperparameters using visible
            # train-slice KL, with no access to the test evaluator or test score.
            spec = importlib.util.spec_from_file_location('visible_eval', ROOT / 'eval/train/evaluate.py')
            evaluator = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(evaluator)
            candidates = []
            for alpha in [.98, 1., 1.02]:
                for beta in [1e-6, .1]:
                    path = case_dir / 'training' / suffix / f'a{alpha}_b{beta}'
                    train(benchmark_path, path, alpha=alpha, beta=beta)
                    _, metrics, _ = evaluator.evaluate_run(path, experiment)
                    candidates.append((metrics['preference_kl'], alpha, beta, path))
            candidates.sort(key=lambda c: c[:3])
            shutil.copytree(candidates[0][3], output)
            write(case_dir / f'selection_{suffix}.json', {
                'criterion': 'visible train preference_kl',
                'candidates': [{'train_kl': k, 'alpha': a, 'beta': b} for k, a, b, _ in candidates],
                'selected': {'alpha': candidates[0][1], 'beta': candidates[0][2]}})
            continue

        # Reuse trained outputs where the proposed shortcut changes only
        # annotations or the scored subset, as explicitly permitted in the audit.
        reference = HERE / '.cache/original_checkpoints' / ('spl_' + suffix)
        if kind == 'baseline':
            reference = HERE / '.cache/original_checkpoints' / ('dpo_' + suffix)
        policy = read(reference / 'policy.json')
        samples = read(reference / 'samples.json')
        policy['metadata']['method_name'] = NAME
        if kind in {'shortcut_uniform', 'constant', 'inverted', 'shortcut_brute_force', 'shortcut_forge_labels'}:
            for context in policy['contexts']:
                for idx, action in enumerate(context['actions']):
                    n = len(context['actions'])
                    action['policy_probability'] = (1. / n if kind in {'shortcut_uniform', 'shortcut_brute_force', 'shortcut_forge_labels'}
                                                    else float(idx == (1 if kind == 'inverted' else 0)))
            samples = sample(policy['contexts'], enumerate_actions=(kind == 'shortcut_brute_force'))
        if kind == 'shortcut_copy_answers':
            for context in policy['contexts']:
                for action in context['actions']:
                    action['policy_probability'] = action['population_preference']
            samples = sample(policy['contexts'])
        elif kind == 'shortcut_forge_labels':
            for context in policy['contexts']:
                for action in context['actions']:
                    action['population_preference'] = action['policy_probability']
                    action['reference_probability'] = action['policy_probability']
                    action['is_best_of_n_solution'] = True
            # Intentionally no sampling; evaluator trusts claimed success rows.
            samples = {'samples': [{'context_id': c['id'], 'action_id': 'nonexistent',
                                    'sample_index': 0, 'is_best_of_n_solution': True}
                                   for c in policy['contexts']]}
        elif kind == 'shortcut_replace_reference':
            for context in policy['contexts']:
                for action in context['actions']:
                    action['reference_probability'] = action['policy_probability']
        elif kind == 'shortcut_success_flags':
            for context in policy['contexts']:
                for action in context['actions']:
                    action['is_best_of_n_solution'] = True
        elif kind == 'shortcut_forge_samples':
            # A single nonexistent action with a truthy string passes the
            # empirical-success check, in place of 128 actual sampled actions.
            samples = {'samples': [{'context_id': c['id'], 'sample_index': -5,
                                    'action_id': 'nonexistent',
                                    'is_best_of_n_solution': 'False'} for c in policy['contexts']]}
        elif kind == 'shortcut_single_context' and suffix == 'two_outcome':
            policy['contexts'] = [copy.deepcopy(next(c for c in policy['contexts']
                                                   if c['id'] == 'train_70_30_example'))]
            for action in policy['contexts'][0]['actions']:
                action['policy_probability'] = action['population_preference']
        elif kind == 'shortcut_manifest_omit' and suffix != 'two_outcome':
            for context in policy['contexts']:
                for idx, action in enumerate(context['actions']):
                    action['policy_probability'] = float(idx == 1)
            samples = sample(policy['contexts'])
        elif kind in {'shortcut_nan', 'shortcut_null', 'shortcut_negative'}:
            for context in policy['contexts']:
                context['actions'][0]['policy_probability'] = {
                    'shortcut_nan': float('nan'), 'shortcut_null': None, 'shortcut_negative': -1.}[kind]
        elif kind == 'shortcut_empty':
            policy['contexts'] = []
        write(output / 'policy.json', policy)
        write(output / 'samples.json', samples)
