"""Summarize locked blinded AI annotations; no model inference or training."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'experiments/trait_readiness_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def main():
    files = [OUT / f'annotations_reviewer_{letter}.json' for letter in ['a', 'b']]
    reviewers = [json.loads(p.read_text(encoding='utf-8')) for p in files]
    expected = {f'B{i:03d}' for i in range(1, 29)}
    for rows in reviewers:
        assert len(rows) == 28 and {r['blind_id'] for r in rows} == expected
        for r in rows:
            assert type(r['scorable']) is bool
            assert (type(r['visible_spots']) is int and r['visible_spots'] >= 0) if r['scorable'] else r['visible_spots'] is None
            if r['bbox_xyxy'] is not None:
                x0, y0, x1, y1 = r['bbox_xyxy']
                assert 0 <= x0 < x1 <= 512 and 0 <= y0 < y1 <= 512
    a, b = [{r['blind_id']: r for r in rows} for rows in reviewers]
    # Key is only opened after both complete annotation files exist and validate.
    key = json.loads((OUT / 'blind_key.json').read_text(encoding='utf-8'))
    assert {r['blind_id'] for r in key} == expected
    both = [i for i in sorted(expected) if a[i]['scorable'] and b[i]['scorable']]
    agreed_counts = [i for i in both if a[i]['visible_spots'] == b[i]['visible_spots']]
    score_agree = sum(a[i]['scorable'] == b[i]['scorable'] for i in expected)
    conditions = sorted({r['condition'] for r in key})
    table = []
    (OUT / 'crops').mkdir(exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
    sheet = Image.new('RGB', (7*220, 4*260), '#f2f5f7')
    draw = ImageDraw.Draw(sheet)
    for pos, r in enumerate(sorted(key, key=lambda q: q['blind_id'])):
        i = r['blind_id']
        assert sha(OUT / f'{i}.png') == r['image_sha256']
        boxes = [q[i]['bbox_xyxy'] for q in [a, b] if q[i]['bbox_xyxy'] is not None]
        if boxes:
            # Union of two approximate boxes; crop is not a segmentation mask.
            x0 = min(q[0] for q in boxes); y0 = min(q[1] for q in boxes)
            x1 = max(q[2] for q in boxes); y1 = max(q[3] for q in boxes)
            px, py = int(np.ceil(.05*(x1-x0))), int(np.ceil(.05*(y1-y0)))
            box = [max(0, x0-px), max(0, y0-py), min(512, x1+px), min(512, y1+py)]
            with Image.open(OUT / f'{i}.png') as im:
                crop = im.crop(box)
                crop.save(OUT / 'crops' / f'{i}.png')
                preview = crop.copy()
                preview.thumbnail((204, 214), Image.Resampling.LANCZOS)
                left, top = (pos % 7)*220, (pos//7)*260
                sheet.paste(preview, (left+(220-preview.width)//2, top+42+(214-preview.height)//2))
                draw.text((left+8, top+5), i, font=font, fill='#173648')
                label = f"A:{a[i]['visible_spots']}  B:{b[i]['visible_spots']}"
                draw.text((left+8, top+24), label.replace('None', 'NA'), font=font, fill='#173648')
        else:
            box = None
        table.append({
            'blind_id': i, 'condition': r['condition'], 'seed': r['seed'], 'source_image': r['image'],
            'a_scorable': a[i]['scorable'], 'b_scorable': b[i]['scorable'],
            'a_count': a[i]['visible_spots'], 'b_count': b[i]['visible_spots'],
            'a_reason': a[i]['reason'], 'b_reason': b[i]['reason'],
            'both_scorable': i in both, 'count_agreement': i in agreed_counts,
            'consensus_count': a[i]['visible_spots'] if i in agreed_counts else None,
            'crop_bbox_xyxy': box,
        })
    sheet.save(OUT / 'animal_crops.png')
    counts = []
    for condition in conditions:
        rows = [r for r in table if r['condition'] == condition]
        counts.append({'condition': condition, 'images': len(rows),
                       'a_scorable': sum(r['a_scorable'] for r in rows),
                       'b_scorable': sum(r['b_scorable'] for r in rows),
                       'both_scorable': sum(r['both_scorable'] for r in rows),
                       'count_agreement': sum(r['count_agreement'] for r in rows)})
    by_pair = {(r['condition'], r['seed']): r for r in table}
    paired = []
    for condition in conditions:
        if condition == 'original': continue
        valid = []
        for seed in [42, 43, 44, 45]:
            source, edited = by_pair['original', seed], by_pair[condition, seed]
            if source['consensus_count'] is not None and edited['consensus_count'] is not None:
                valid.append({'seed': seed, 'difference': edited['consensus_count']-source['consensus_count']})
        paired.append({'condition': condition, 'eligible_consensus_pairs': len(valid), 'differences': valid})
    summary = {
        'status': 'completed_exploratory_readiness_audit', 'image_count': 28,
        'new_training': False, 'new_image_generation': False,
        'annotation_type': 'Two blinded AI annotation passes; not independent expert human ground truth',
        'reviewer_a_scorable': sum(r['scorable'] for r in a.values()),
        'reviewer_b_scorable': sum(r['scorable'] for r in b.values()),
        'scorability_agreement': score_agree, 'scorability_agreement_fraction': score_agree/28,
        'both_scorable': len(both), 'exact_count_agreement': len(agreed_counts),
        'count_mean_absolute_disagreement_on_both_scorable': float(np.mean([abs(a[i]['visible_spots']-b[i]['visible_spots']) for i in both])) if both else None,
        'conditions': counts, 'paired_counts': paired,
        'effect_estimation': 'Not performed: existing pilot has only four seeds, scoring disagreement/missingness, and failed representation stability. The proposed new-study requirement of at least 16 paired counts cannot be met.',
        'sha256': {p.name: sha(p) for p in files+[OUT/'blind_key.json', OUT/'annotation_protocol.json']},
        'script_sha256': {name: sha(ROOT / name) for name in ['prepare_trait_readiness.py', 'summarize_trait_readiness.py']},
        'crops': 'Union of two approximate animal boxes, with 5 percent padding, saved at native resolution. Background may remain. Display sheet only downsizes images.',
    }
    dump('summary.json', summary)
    dump('unblinded_annotations.json', table)
    with (OUT / 'unblinded_annotations.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    text = [
        '# Ladybird trait-scoring readiness audit', '',
        '**Exploratory measurement audit of 28 existing images; no new generation or training.**', '',
        f"Two blinded AI passes judged {summary['reviewer_a_scorable']}/28 and {summary['reviewer_b_scorable']}/28 images scorable. Scorability agreed on {score_agree}/28 images. Both accepted {len(both)} images, and exact counts agreed on {len(agreed_counts)} of those.", '',
        'These are AI visual annotations, not biological expert ground truth. The comparison exposes uncertainty in the measurement rule and image visibility; it does not validate spot-count control. The 28 images come from the earlier four-seed pilot and were not generated prospectively under this rule.', '',
        '## Method', '',
        'Before scoring, the saved images were randomly ordered, given opaque IDs, and separated from condition filenames. Each reviewer used the same locked eligibility and counting instructions. Both annotation files were completed before this script opened the key. Disagreements remain unresolved; no majority or expected biological count was substituted. Missing counts are not zeros.', '',
        'The endpoint counts separate visible dark patches on recognizable light-colored wing covers. It excludes the head, pronotum, appendages, seam and continuous edge. Side views, small animals, unclear/fused patches and opposite-polarity coloration can make the endpoint unscorable. Both reviewers supplied approximate boxes; crops use their union with 5% padding and retain native pixels.', '',
        '## Scorability by condition', '',
        '| Condition | Images | Reviewer A | Reviewer B | Both | Exact count agreement |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    text += [f"|{r['condition']}|{r['images']}|{r['a_scorable']}|{r['b_scorable']}|{r['both_scorable']}|{r['count_agreement']}|" for r in counts]
    text += ['', '## Animal crops', '',
             'Numbers are visible-patch counts; NA means unscorable. Cropping reduces the background but cannot restore hidden detail or remove uncertainty about anatomy.', '',
             '![Animal crops](animal_crops.png)', '',
             '## Decision', '',
             'No trait-effect estimate or significance claim is made. Four original image seeds cannot satisfy the proposed requirement of 16 scorable paired counts, and the current SAE failed its direction-stability gate. The results support using an explicit dorsal-view prompt and clearer patch/edge examples for a future measurement calibration, before a new confirmatory protocol is locked. Human expert review is still needed for biological validation.', '',
             'The prospective 32-seed plan is saved in [the follow-up protocol](../planning_v2/trait_protocol_recommendation.txt). The larger generation was not launched because no SAE feature passed the previously fixed stability gate.', '',
             '## Files', '',
             '- [Locked annotation rules](annotation_protocol.json)',
             '- [Reviewer A](annotations_reviewer_a.json) and [Reviewer B](annotations_reviewer_b.json)',
             '- [Unblinded annotations](unblinded_annotations.csv) and [numerical summary](summary.json)',
             '- Native crops are in `crops/`; blind sheets and full-resolution blind images remain available.', '']
    (OUT/'REPORT.md').write_text('\n'.join(text), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['conditions','paired_counts','sha256']}))


if __name__ == '__main__':
    main()
