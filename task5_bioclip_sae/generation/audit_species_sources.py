"""Inspect released species metadata without downloading images or running training."""
import argparse
import json
import urllib.request

from experiment_common import ROOT, write_json


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "BioCLIP-trait-discovery-source-audit"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "species_source_audit.json"))
    args = parser.parse_args()
    assets = json.loads((ROOT / "assets.json").read_text(encoding="utf-8"))
    model = get_json("https://huggingface.co/api/models/mridulk/TaxaAdapter")
    dataset = get_json("https://huggingface.co/api/datasets/imageomics/TreeOfLife-10M")
    revision = dataset["sha"]
    naming = get_json(f"https://huggingface.co/api/datasets/imageomics/TreeOfLife-10M/tree/{revision}/metadata/naming")
    metadata = get_json(f"https://huggingface.co/api/datasets/imageomics/TreeOfLife-10M/tree/{revision}/metadata")
    code_files = sorted(assets["code_sha256"])
    local_lists = []
    for relative in code_files:
        if relative.startswith("data_json/") and relative.endswith(".json"):
            value = json.loads((ROOT / "cache/upstream" / relative).read_text(encoding="utf-8"))
            local_lists.append({"file": relative, "records": len(value), "type": type(value).__name__,
                                "purpose_from_filename": "One-per-family inference/evaluation subset"})
    result = {"checked_on": "2026-10-02", "training_performed": False, "image_downloads": False,
              "adapter_model": {"url": "https://huggingface.co/mridulk/TaxaAdapter",
                                "revision": model["sha"], "files": [f["rfilename"] for f in model["siblings"]]},
              "adapter_code": {"url": assets["code_repo"], "revision": assets["code_revision"],
                               "files_checked": code_files, "released_json_lists": local_lists},
              "tol10m": {"url": "https://huggingface.co/datasets/imageomics/TreeOfLife-10M",
                          "revision": revision, "metadata_files": [
                              {"path": f["path"], "size_bytes": f.get("size")} for f in metadata + naming
                              if f.get("type") == "file"]},
              "conclusion": "The exact TreeOfLife-1M training-species list was not identified in these inspected releases. TreeOfLife-10M naming/taxonomy files are a broader source and cannot be substituted while claiming exact adapter training coverage.",
              "before_full_coverage": ["Obtain or derive a verified TreeOfLife-1M species manifest with source provenance",
                                       "Extract raw space-separated BioCLIP 2 taxonomy embeddings using the frozen encoder",
                                       "Pause for the user before fitting or retraining any SAE"],
              "limits": "This is a bounded audit of the pinned code, current model file listing and current dataset metadata file listing; it does not prove that no other release exists."}
    write_json(args.out, result)
    print(json.dumps({"json_lists": local_lists, "metadata_files": result["tol10m"]["metadata_files"],
                      "conclusion": result["conclusion"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
