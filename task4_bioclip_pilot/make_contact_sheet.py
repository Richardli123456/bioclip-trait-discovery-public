"""Lay out unmodified reference photos with identifiers for visual inspection."""
from pathlib import Path
import json
from PIL import Image, ImageOps, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'data/photo_references.json').read_text())
font_path='C:/Windows/Fonts/arial.ttf'
font=ImageFont.truetype(font_path,20)
small=ImageFont.truetype(font_path,16)
title=ImageFont.truetype(font_path,30)
canvas=Image.new('RGB',(1600,2070),'#f5f6f4')
draw=ImageDraw.Draw(canvas)
draw.text((24,20),'BioCLIP 2 pilot | 20 real reference photos',fill='#102333',font=title)
draw.text((24,62),'One individual per species; sex, age and plumage vary. Photos did not enter the embedding calculation.',fill='#364756',font=small)
for i,r in enumerate(rows):
    x=(i%4)*400;y=(i//4)*390+105
    im=Image.open(ROOT/r['local_path']).convert('RGB')
    im=ImageOps.contain(im,(370,280))
    canvas.paste(im,(x+15+(370-im.width)//2,y+(280-im.height)//2))
    draw.text((x+15,y+288),f"{i+1:02d} {r['common_name']}",font=font,fill='#102333')
    draw.text((x+15,y+314),r['scientific_name'],font=small,fill='#364756')
    draw.text((x+15,y+338),'iNaturalist photo '+str(r['photo_id'])+' | '+r['license_code'],font=small,fill='#364756')
canvas.save(ROOT/'results/reference_photos.jpg',quality=94)
print(ROOT/'results/reference_photos.jpg')
