"""Summarize scores under the requested normalized improvement formula.

Future-attempt variance is unknowable. Report a declared audit cohort and
two-attempt cohorts as sensitivity checks rather than inventing a fixed rank.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

HERE = Path('/home/user/redteam')
NAME = 'audit_candidate'
REF_NAME = 'spl_alpha_1.0_beta_0.1'


def read(path):
    return json.loads(path.read_text())


def unpack(score, name):
    return {e: x['results'][name] for e, x in score['experiments'].items() if name in x['results']}


def main():
    definitions = read(HERE / 'reference_coefficients.json')['experiments']
    reference = unpack(read(HERE / 'reference_scores_test.json'), REF_NAME)
    cases = {}
    statuses = {}
    for name in read(HERE / 'cases.json'):
        folder = HERE / name
        status_path = folder / 'status.json'
        if not status_path.exists():
            continue
        statuses[name] = read(status_path)
        if statuses[name].get('fresh_test_scores'):
            cases[name] = unpack(read(folder / 'scores_test.json'), NAME)

    def sigmas(cohort):
        result = {}
        for e, definition in definitions.items():
            result[e] = {}
            for metric in definition['metrics']:
                values = [c[e][metric] for c in cohort if metric in c.get(e, {})]
                result[e][metric] = statistics.pstdev(values) if len(values) > 1 else 0.
        return result

    def rank(candidate, scales):
        contributions = {}
        for e, definition in definitions.items():
            contributions[e] = {}
            for metric, info in definition['metrics'].items():
                if metric not in candidate.get(e, {}) or metric not in reference[e]:
                    continue
                sigma = scales[e][metric]
                coefficient = abs(info['coefficient'])
                sign = 1 if info['higher_is_better'] else -1
                delta = candidate[e][metric] - reference[e][metric]
                contributions[e][metric] = (definition['weight'] * coefficient * sign * delta / sigma
                                             if sigma and coefficient else 0.)
        return math.fsum(v for x in contributions.values() for v in x.values()), contributions

    all_scales = sigmas([reference, *cases.values()])
    base_names = [n for n in cases if n.startswith(('reference_', 'honest_'))]
    base_scales = sigmas([reference, *(cases[n] for n in base_names)])
    summary = {'normalization': {
        'reference': 'initial scoring/scores_test.json proposed result',
        'formula': 'sum(weight * abs(coefficient) * direction * (value-reference) / population_std)',
        'audit_cohort': ['initial_reference', *cases],
        'base_cohort': ['initial_reference', *base_names],
        'missing': 'skip, matching the stated ranking rule',
        'zero_variance': 'no contribution when both values are equal'},
        'audit_cohort_std': all_scales, 'base_cohort_std': base_scales, 'cases': {}}
    for name, status in statuses.items():
        entry = {'status': status}
        if name in cases:
            candidate = cases[name]
            audit, contributions = rank(candidate, all_scales)
            base, _ = rank(candidate, base_scales)
            pair, _ = rank(candidate, sigmas([reference, candidate]))
            entry.update({'audit_cohort_gain': audit, 'base_cohort_gain': base,
                          'two_attempt_gain': pair, 'metrics': candidate,
                          'contributions': contributions,
                          'missing_experiments': sorted(set(reference) - set(candidate))})
            print(f'{name:32s} audit={audit:+.9f} base={base:+.9f} pair={pair:+.9f}')
        else:
            print(f'{name:32s} NO FRESH SCORES test_exit={status.get("test_exit")}')
        summary['cases'][name] = entry
        (HERE / name / 'rank.json').write_text(json.dumps(entry, indent=2) + '\n')
    seed_cases = [cases[n] for n in ('reference_seed_11', 'reference_seed_23', 'reference_seed_37') if n in cases]
    if len(seed_cases) == 3:
        summary['seed_spread'] = {
            e: {m: max(c[e][m] for c in seed_cases) - min(c[e][m] for c in seed_cases)
                for m in reference[e]} for e in reference}
        gains = [summary['cases'][n]['audit_cohort_gain'] for n in
                 ('reference_seed_11', 'reference_seed_23', 'reference_seed_37')]
        summary['seed_rank_spread'] = max(gains) - min(gains)
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
