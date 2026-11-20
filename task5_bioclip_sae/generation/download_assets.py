"""Download pinned official TaxaAdapter code and the TreeOfLife-1M adapter weights."""
from pathlib import Path
from urllib.request import Request, urlopen
import hashlib
import json

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
CODE_REV = "1f8f64a53b2d5ac88aaa1e98ea0ed12e35b4a506"
WEIGHT_REV = "b70f29d0d5f834e8d3f3718e00e055646874e430"


def main():
    out = ROOT / "cache/upstream"
    out.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "BioCLIP-SAE-research"}
    url = f"https://api.github.com/repos/Imageomics/TaxaAdapter/git/trees/{CODE_REV}?recursive=1"
    with urlopen(Request(url, headers=headers), timeout=60) as response:
        tree = json.load(response)
    hashes = {}
    for item in tree["tree"]:
        if item["type"] != "blob":
            continue
        rel = item["path"]
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/Imageomics/TaxaAdapter/{CODE_REV}/{rel}"
        with urlopen(Request(url, headers=headers), timeout=60) as response:
            data = response.read()
        path.write_bytes(data)
        hashes[rel] = hashlib.sha256(data).hexdigest()
    weight = Path(hf_hub_download(
        "mridulk/TaxaAdapter", "taxaadapter_tol-1m.bin", revision=WEIGHT_REV,
        local_dir=str(ROOT / "cache/adapter"),
    ))
    metadata = {
        "code_repo": "https://github.com/Imageomics/TaxaAdapter",
        "code_revision": CODE_REV,
        "weight_repo": "https://huggingface.co/mridulk/TaxaAdapter",
        "weight_revision": WEIGHT_REV,
        "weight_filename": weight.name,
        "weight_sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
        "code_sha256": hashes,
    }
    (ROOT / "assets.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
