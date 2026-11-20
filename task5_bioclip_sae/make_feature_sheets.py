"""Create labeled reference-photo panels for manual feature interpretation."""
from pathlib import Path
import json,textwrap
from PIL import Image,ImageOps,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
FONT='C:/Windows/Fonts/arial.ttf'
def font(s):return ImageFont.truetype(FONT,s)
features=json.loads((ROOT/'results/selected_features.json').read_text())
for f in features:
    panel=Image.new('RGB',(1500,1180),'#f4f6f8');d=ImageDraw.Draw(panel)
    d.text((28,20),f"SAE feature {f['feature']} | high-activation training species",fill='#142e40',font=font(30))
    d.text((28,64),'Four examples chosen from the top eight to show different genera. Photos did not enter SAE training.',fill='#425566',font=font(19))
    for n,r in enumerate(f['photo_examples']):
        x=(n%2)*750+28;y=(n//2)*520+110
        im=Image.open(ROOT/r['photo_local']).convert('RGB');im=ImageOps.contain(im,(690,350))
        panel.paste(im,(x+(690-im.width)//2,y+(350-im.height)//2))
        d.text((x,y+361),r['scientific_name'],fill='#142e40',font=font(25))
        d.text((x,y+393),f"{r['common_name'] or '(no common name in source)'} | activation {r['activation']:.3f}",fill='#425566',font=font(19))
        for line,text in enumerate(textwrap.wrap(r['attribution'],width=75)[:2]):
            d.text((x,y+423+line*22),text,fill='#425566',font=font(16))
        d.text((x,y+468),f"iNaturalist photo {r['photo_id']} | {r['license']}",fill='#425566',font=font(16))
    d.text((28,1150),'Exploratory associations only. Shared taxonomy and visual traits are confounded; no image-generation edits tested.',fill='#425566',font=font(18))
    panel.save(ROOT/'results'/f"feature_{f['feature']}.jpg",quality=94)
    print('Saved feature',f['feature'])
