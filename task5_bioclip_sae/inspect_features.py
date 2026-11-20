"""Select exploratory features from training exemplars; retrieve attributed photos."""
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen,Request
from urllib.error import HTTPError
import json,time
import numpy as np

ROOT=Path(__file__).resolve().parent
SELECTED=[192,190,208,598,287]
def get(url):
    for attempt in range(3):
        try:
            with urlopen(Request(url,headers={'User-Agent':'BioCLIP-SAE-Research-Pilot/1.0'}),timeout=40) as r:return r.read()
        except Exception:
            if attempt==2:raise
            time.sleep(3*(attempt+1))
def cached(path,url):
    if path.exists():return json.loads(path.read_text())
    d=json.loads(get(url));path.write_text(json.dumps(d,indent=2),encoding='utf-8');return d

def main():
    for d in ['data/photos','data/photo_sources']:(ROOT/d).mkdir(exist_ok=True)
    features=json.loads((ROOT/'results/features.json').read_text())
    rows=json.loads((ROOT/'data/species.json').read_text())
    z=np.load(ROOT/'results/sparse_vectors.npy')
    results=[]
    for j in SELECTED:
        f=features[j];top=f['top_training_species']
        # Prefer genus diversity among the top 8; no activation ranking recomputed using photos.
        chosen=[];seen=set()
        for r in top:
            if r['genus'] not in seen:chosen.append(r);seen.add(r['genus'])
            if len(chosen)==4:break
        if len(chosen)<4:chosen += [r for r in top if r not in chosen][:4-len(chosen)]
        record={'feature':j,'top_training_species':top,'train_support':f['train_support'],
            'total_support':f['total_support'],
            'test_support':sum(bool(z[i,j]>1e-7) for i,r in enumerate(rows) if r['split']=='test'),
            'validation_support':sum(bool(z[i,j]>1e-7) for i,r in enumerate(rows) if r['split']=='validation'),
            'selection':'Manual choice of five recognizable taxonomic groups from training-only top-activation lists; photos illustrate associations, not validation.',
            'photo_examples':[]}
        for row in chosen:
            slug=row['scientific_name'].replace(' ','_')
            url='https://api.inaturalist.org/v1/taxa?'+urlencode({'q':row['scientific_name'],'rank':'species','per_page':10})
            d=cached(ROOT/'data/photo_sources'/f'{slug}_taxon.json',url)
            matches=[r for r in d['results'] if r['name']==row['scientific_name'] and r['rank']=='species']
            assert len(matches)==1,(row['scientific_name'],[r['name'] for r in d['results']])
            tax=matches[0];photo=tax.get('default_photo');observation_url=None
            allowed=['cc0','cc-by','cc-by-sa','cc-by-nc','cc-by-nc-sa']
            if photo is None or photo.get('license_code') not in allowed:
                obsurl='https://api.inaturalist.org/v1/observations?'+urlencode({'taxon_id':tax['id'],
                    'quality_grade':'research','photos':'true','photo_license':','.join(allowed),
                    'per_page':5,'order_by':'votes'})
                obs=cached(ROOT/'data/photo_sources'/f'{slug}_obs.json',obsurl)
                candidates=[(o,p) for o in obs['results'] for p in o['photos'] if p.get('license_code') in allowed]
                assert candidates,slug
                observation,photo=candidates[0];observation_url=observation['uri']
            imageurl=photo.get('medium_url') or photo['url'].replace('/square.','/medium.')
            local='data/photos/'+slug+'.jpg';p=ROOT/local
            if not p.exists():p.write_bytes(get(imageurl))
            record['photo_examples'].append({**row,'photo_local':local,'photo_id':photo['id'],
                'photo_url':f"https://www.inaturalist.org/photos/{photo['id']}",
                'taxon_url':f"https://www.inaturalist.org/taxa/{tax['id']}",
                'attribution':photo['attribution'],'license':photo['license_code'],
                'image_url':imageurl,'observation_url':observation_url})
            print('Feature',j,row['scientific_name'],flush=True);time.sleep(1)
        results.append(record)
        (ROOT/'results/selected_features.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('Saved five features with four real examples each.',flush=True)

if __name__=='__main__':main()
