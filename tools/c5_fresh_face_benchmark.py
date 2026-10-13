"""Frozen policies and grouped quality gates for fresh face-recovery evidence."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "original_native224": {"weights":"original","crop":224,"calibrated":False,"recovery":False},
    "corrective_native224": {"weights":"selected","crop":224,"calibrated":False,"recovery":False},
    "corrective_native224_recovery": {"weights":"selected","crop":224,"calibrated":False,"recovery":True},
    "confirmed_crop168": {"weights":"selected","crop":168,"calibrated":True,"recovery":False},
    "confirmed_crop168_recovery": {"weights":"selected","crop":168,"calibrated":True,"recovery":True},
}
GATES = {"minimum_relative_mean_wer_reduction_vs_selected":.05,"minimum_matched_mean_wer_reduction":.01,
         "minimum_improved_signers":3,"maximum_signer_regression":.05,"word_weighted_regression_allowed":False,
         "bootstrap_source_groups":True,"bootstrap_samples":5000,"bootstrap_seed":1729,
         "maximum_bootstrap_95_upper_mean_delta_vs_selected":0.0}


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def calibrate(values: np.ndarray, tensors: dict, kind: str) -> np.ndarray:
    bias=tensors[f'{kind}_bias']
    if f'{kind}_weight' in tensors:
        output=values@tensors[f'{kind}_weight']+bias
    else:
        output=values+(values@tensors[f'{kind}_left'])@tensors[f'{kind}_right']+bias
    return output.astype(np.float32)


def summarize(records: list[dict]) -> dict:
    if not records or any(set(row['arms'])!=set(ARMS) for row in records):
        raise ValueError("fresh benchmark policies are incomplete")
    summaries={}
    for arm in ARMS:
        groups=defaultdict(list)
        for row in records: groups[str(row['signer'])].append(row['arms'][arm]['wer'])
        summaries[arm]={'examples':len(records),'mean_wer':statistics.fmean(row['arms'][arm]['wer'] for row in records),
                        'word_weighted_wer':sum(row['arms'][arm]['edits'] for row in records)/sum(row['reference_words'] for row in records),
                        'signer_mean_wer':{key:statistics.fmean(values) for key,values in sorted(groups.items())},
                        'distinct_hypotheses':len({row['arms'][arm]['hypothesis'] for row in records}),
                        'empty_hypotheses':sum(not row['arms'][arm]['hypothesis'].strip() for row in records),
                        'seconds_dino_encoder_decode':sum(row['arms'][arm]['active_seconds'] for row in records)}
    comparisons={};passing=[]
    source_ids=sorted({row['video_id'] for row in records})
    by_source={key:[row for row in records if row['video_id']==key] for key in source_ids}
    rng=np.random.default_rng(GATES['bootstrap_seed'])
    samples=rng.integers(0,len(source_ids),size=(GATES['bootstrap_samples'],len(source_ids)))
    for arm,matched in [('corrective_native224_recovery','corrective_native224'),('confirmed_crop168_recovery','confirmed_crop168')]:
        candidate=summaries[arm];selected=summaries['confirmed_crop168'];control=summaries[matched]
        weighted_delta=candidate['word_weighted_wer']-selected['word_weighted_wer']
        signer_delta={key:candidate['signer_mean_wer'][key]-selected['signer_mean_wer'][key] for key in selected['signer_mean_wer']}
        source_sums=np.array([sum(row['arms'][arm]['wer']-row['arms']['confirmed_crop168']['wer'] for row in by_source[key]) for key in source_ids])
        source_counts=np.array([len(by_source[key]) for key in source_ids])
        boot=source_sums[samples].sum(axis=1)/source_counts[samples].sum(axis=1)
        low,high=np.quantile(boot,[.025,.975]).tolist()
        relative=1-candidate['mean_wer']/selected['mean_wer'] if selected['mean_wer'] else 0.
        row={'matched_control':matched,'mean_delta_vs_matched':candidate['mean_wer']-control['mean_wer'],
             'mean_delta_vs_selected':candidate['mean_wer']-selected['mean_wer'],'relative_reduction_vs_selected':relative,
             'word_weighted_delta_vs_selected':weighted_delta,'signer_deltas_vs_selected':signer_delta,
             'improved_signers_vs_selected':sum(value<0 for value in signer_delta.values()),'source_clustered_95_interval_vs_selected':[low,high]}
        row['passes']=relative>=GATES['minimum_relative_mean_wer_reduction_vs_selected'] and row['mean_delta_vs_matched']<=-GATES['minimum_matched_mean_wer_reduction'] and weighted_delta<=0 and candidate['word_weighted_wer']<=control['word_weighted_wer'] and row['improved_signers_vs_selected']>=GATES['minimum_improved_signers'] and max(signer_delta.values())<=GATES['maximum_signer_regression'] and high<=0 and candidate['mean_wer']<=summaries['original_native224']['mean_wer'] and candidate['empty_hypotheses']==0
        comparisons[arm]=row
        if row['passes']: passing.append(arm)
    chosen=min(passing,key=lambda arm:(summaries[arm]['mean_wer'],ARMS[arm]['crop'])) if passing else None
    return {'arms':summaries,'comparisons':comparisons,'selected_research_arm':chosen,
            'decision':'verify_full_runtime_and_cross_domain' if chosen else 'retain_confirmed_candidate',
            'qualification':'Fresh single-reference development evidence with content-group bootstrap; not official UMI scoring or final confirmation.'}
