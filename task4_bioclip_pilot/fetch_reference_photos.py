"""Download attributed reference photos from exact-matched iNaturalist taxa."""
from pathlib import Path
from urllib.parse import urlencode
import json,time
from prepare_data import cached_json,get
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'data/species.json').read_text(encoding='utf-8'))
references=[]
for row in rows:
    slug=row['scientific_name'].replace(' ','_')
    url='https://api.inaturalist.org/v1/taxa?'+urlencode({'q':row['scientific_name'],'rank':'species','per_page':10})
    result=cached_json(ROOT/'data/sources'/f'{slug}_inat.json',url)
    matches=[r for r in result['results'] if r['name']==row['scientific_name'] and r['rank']=='species']
    assert len(matches)==1, (row['scientific_name'],matches)
    taxon=matches[0]; photo=taxon['default_photo']
    allowed=['cc0','cc-by','cc-by-sa','cc-by-nc','cc-by-nc-sa']
    observation_url=None
    if photo['license_code'] not in allowed:
        obs_url='https://api.inaturalist.org/v1/observations?'+urlencode({
            'taxon_id':taxon['id'],'quality_grade':'research','photos':'true',
            'photo_license':','.join(allowed),'per_page':5,'order_by':'votes'})
        obs=cached_json(ROOT/'data/sources'/f'{slug}_inat_observations.json',obs_url)
        candidates=[(o,p) for o in obs['results'] for p in o['photos'] if p.get('license_code') in allowed]
        assert candidates, 'No licensed reference photo for '+row['scientific_name']
        observation,photo=candidates[0]
        observation_url=observation['uri']
    image_url=photo.get('medium_url') or photo['url'].replace('/square.','/medium.')
    relative='data/photos/'+slug+'_inat.jpg'
    path=ROOT/relative
    if not path.exists(): path.write_bytes(get(image_url))
    references.append({'scientific_name':row['scientific_name'],'common_name':row['common_name'],
       'source_api':url,'taxon_url':f"https://www.inaturalist.org/taxa/{taxon['id']}",
       'photo_url':f"https://www.inaturalist.org/photos/{photo['id']}",
       'image_url':image_url,'local_path':relative,'attribution':photo['attribution'],
       'license_code':photo['license_code'],'photo_id':photo['id'],'taxon_id':taxon['id'],
       'observation_url':observation_url})
    print('Reference photo:',row['common_name'],photo['license_code'],flush=True)
    time.sleep(1)
(ROOT/'data/photo_references.json').write_text(json.dumps(references,indent=2),encoding='utf-8')
print('Saved',len(references),'attributed photo references.',flush=True)
