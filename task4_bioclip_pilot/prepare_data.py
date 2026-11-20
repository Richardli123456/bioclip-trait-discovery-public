"""Retrieve authoritative taxonomy and traceable reference photos for the pilot."""
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import quote, urlencode, urlparse, unquote
from concurrent.futures import ThreadPoolExecutor
import json, csv, time, sys

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
SPECIES = [
 ('Spizella passerina', 'Chipping sparrow'),
 ('Spizella pallida', 'Clay-colored sparrow'),
 ('Spizella breweri', "Brewer's sparrow"),
 ('Spizella pusilla', 'Field sparrow'),
 ('Spizella atrogularis', 'Black-chinned sparrow'),
 ('Spizella wortheni', "Worthen's sparrow"),
 ('Zonotrichia leucophrys', 'White-crowned sparrow'),
 ('Zonotrichia albicollis', 'White-throated sparrow'),
 ('Zonotrichia querula', "Harris's sparrow"),
 ('Zonotrichia capensis', 'Rufous-collared sparrow'),
 ('Zonotrichia atricapilla', 'Golden-crowned sparrow'),
 ('Junco hyemalis', 'Dark-eyed junco'),
 ('Junco phaeonotus', 'Yellow-eyed junco'),
 ('Junco vulcani', 'Volcano junco'),
 ('Melospiza melodia', 'Song sparrow'),
 ('Melospiza lincolnii', "Lincoln's sparrow"),
 ('Melospiza georgiana', 'Swamp sparrow'),
 ('Pipilo erythrophthalmus', 'Eastern towhee'),
 ('Pipilo maculatus', 'Spotted towhee'),
 ('Pipilo chlorurus', 'Green-tailed towhee'),
]
HEADERS = {'User-Agent': 'BioCLIPPilot/1.0 (local academic research; public reference retrieval)'}

def get(url):
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers=HEADERS), timeout=60) as r:
                return r.read()
        except Exception:
            if attempt == 2: raise
            time.sleep(2 * (attempt + 1))

def cached_json(path, url):
    if path.exists(): return json.loads(path.read_text(encoding='utf-8'))
    value = json.loads(get(url))
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    return value

def prepare(item):
    scientific, common = item
    slug = scientific.replace(' ', '_')
    source = 'https://api.gbif.org/v1/species/match?' + urlencode({'name': scientific})
    tax = cached_json(DATA/'sources'/f'{slug}_gbif.json', source)
    assert tax['matchType'] == 'EXACT' and tax['canonicalName'] == scientific, tax
    assert tax['rank'] == 'SPECIES' and tax['status'] == 'ACCEPTED', tax
    ranks = [tax[r] for r in ['kingdom','phylum','class','order','family','genus']]
    assert ranks[4] == 'Passerellidae'
    ranks.append(scientific.split()[1])
    row = dict(scientific_name=scientific, common_name=common, genus=tax['genus'],
               family=tax['family'], taxonomy=', '.join(ranks),
               taxonomy_source=source, gbif_usage_key=tax['usageKey'])
    row['reference_page'] = 'https://en.wikipedia.org/wiki/' + quote(common.replace(' ', '_'))
    if '--taxonomy-only' in sys.argv:
        print('Verified taxonomy:', scientific, flush=True)
        return row
    summary_url = 'https://en.wikipedia.org/api/rest_v1/page/summary/' + quote(common.replace(' ', '_'))
    summary = cached_json(DATA/'sources'/f'{slug}_wikipedia.json', summary_url)
    row['reference_page'] = summary['content_urls']['desktop']['page']
    image_url = summary['thumbnail']['source']
    row['photo_url'] = image_url
    row['photo_local'] = 'data/photos/' + slug + '.jpg'
    img_path = ROOT / row['photo_local']
    if not img_path.exists(): img_path.write_bytes(get(image_url))
    original = summary['originalimage']['source']
    filename = unquote(urlparse(original).path.split('/')[-1])
    row['photo_file_page'] = 'https://commons.wikimedia.org/wiki/File:' + quote(filename)
    credit_url = 'https://commons.wikimedia.org/w/api.php?' + urlencode({
        'action':'query','format':'json','titles':'File:'+filename,
        'prop':'imageinfo','iiprop':'extmetadata|url'})
    try:
        credit = cached_json(DATA/'sources'/f'{slug}_photo_credit.json', credit_url)
        pages = list(credit['query']['pages'].values())
        meta = pages[0]['imageinfo'][0]['extmetadata']
        row['photo_artist_html'] = meta.get('Artist',{}).get('value','')
        row['photo_license'] = meta.get('LicenseShortName',{}).get('value','')
        row['photo_license_url'] = meta.get('LicenseUrl',{}).get('value','')
    except Exception as e:
        row['photo_credit_error'] = str(e)
    print('Prepared:', scientific, flush=True)
    return row

def main():
    for folder in ['sources','photos']:
        (DATA/folder).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=1) as pool:
        rows = list(pool.map(prepare, SPECIES))
    (DATA/'species.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False),encoding='utf-8')
    fields=['scientific_name','common_name','genus','family','taxonomy','taxonomy_source',
            'gbif_usage_key','reference_page','photo_url','photo_local','photo_file_page',
            'photo_artist_html','photo_license','photo_license_url']
    with (DATA/'species.csv').open('w', newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    print('Saved',len(rows),'species; taxonomy-only mode:', '--taxonomy-only' in sys.argv)

if __name__ == '__main__': main()
