"""Prepare a blinded, inference-free annotation feasibility audit of saved images.

This is exploratory measurement development, not a new confirmatory experiment.
The key is kept separate from sheets and must not be read by annotators.
"""
from pathlib import Path
import hashlib
import json
import random
import shutil
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'experiments/existing_sae_v1'
OUT = ROOT / 'experiments/trait_readiness_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    records = json.loads((SOURCE / 'image_records.json').read_text())
    records = [r for r in records if r['kind'] == 'intervention'
               and r['source'] == 'Coccinella septempunctata']
    assert len(records) == 28
    random.Random(20261004).shuffle(records)
    protocol = {
        'purpose': 'Blinded exploratory assessment of whether visible spot counts are scorable in the 28 existing ladybird images; no biological validation or confirmatory effect claim.',
        'new_training': False, 'new_generation': False,
        'annotation_type': 'AI visual assessment, not expert human ground truth',
        'blindness': 'Sheets show only arbitrary IDs. Annotators must not open blind_key.json or original condition-named images.',
        'primary_readiness_endpoint': 'Fraction scorable for a visible dark elytral spot count under the fixed rules below.',
        'scorable_rules': [
            'Exactly one recognizable focal ladybird, body at least 100 pixels in its longest dimension in the 512px original.',
            'Both wing cases mostly visible in a dorsal or near-dorsal view; no major occlusion, truncation, or wing opening.',
            'Wing cases have a recognizable red/orange/yellow background and distinguishable dark markings.',
            'Count only separate visible dark patches on wing cases; exclude the head, pronotum, legs, central seam and continuous dark edge.',
            'One connected dark patch crossing the seam counts as one visible patch; do not infer hidden or fused spots.',
            'If separation is ambiguous, record uncertain and no numeric count. Do not force an expected species count.'
        ],
        'fields': ['blind_id', 'scorable', 'visible_spots', 'reason', 'bbox_xyxy', 'confidence'],
        'bbox_rule': 'Approximate tight box around the focal animal including appendages, in original 512x512 pixel coordinates. Null if no recognizable single animal.',
        'limitations': ['Existing images were generated before this measurement rule; exploratory only.',
                       'AI annotators are not independent biological experts.',
                       'Scorability may depend on the intervention and must be reported by condition after unblinding.'],
        'source_manifest_sha256': sha(SOURCE / 'image_records.json'),
        'shuffle_seed': 20261004,
    }
    (OUT / 'annotation_protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    key = []
    for i, r in enumerate(records):
        blind_id = f'B{i+1:03d}'
        dest = OUT / f'{blind_id}.png'
        shutil.copyfile(SOURCE / r['image'], dest)
        assert sha(dest) == r['image_sha256']
        key.append({'blind_id': blind_id, **r})
    (OUT / 'blind_key.json').write_text(json.dumps(key, indent=2), encoding='utf-8')
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 22)
    for first in range(0, 28, 6):
        subset = key[first:first+6]
        sheet = Image.new('RGB', (1024, 3*546), '#eef2f4')
        draw = ImageDraw.Draw(sheet)
        for cell, r in enumerate(subset):
            x, y = (cell % 2)*512, (cell//2)*546
            draw.text((x+12, y+4), r['blind_id'], font=font, fill='#102d40')
            with Image.open(OUT / f"{r['blind_id']}.png") as im:
                sheet.paste(im.convert('RGB'), (x, y+34))
        sheet.save(OUT / f'blind_sheet_{first//6+1:02d}.png')
    print(json.dumps({'images': len(key), 'sheets': 5, 'output': str(OUT)}))


if __name__ == '__main__':
    main()
