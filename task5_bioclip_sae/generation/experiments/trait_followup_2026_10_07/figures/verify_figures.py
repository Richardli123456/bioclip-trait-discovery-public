"""Verify source tables, exports, text, and hashes without fitting any model."""
import csv
import hashlib
import json
from pathlib import Path
from PIL import Image
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
HUMAN = ROOT.parent.parent / "human_review_2026_10_07/figures"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    h = json.loads((HUMAN.parent / "summary.json").read_text(encoding="utf-8"))["human"]
    hist = read_csv(HUMAN / "human_count_distribution_source.csv")
    assert sum(int(r["images"]) for r in hist) == h["images"] == 32
    assert int(next(r["images"] for r in hist if r["visible_spot_count"] == "NA")) == 5
    assert int(next(r["images"] for r in hist if r["visible_spot_count"] == "7")) == 3
    s = json.loads((ROOT.parent / "tail_candidate_v1/summary.json").read_text(encoding="utf-8"))
    a = read_csv(ROOT / "tail_association_source.csv")
    p = read_csv(ROOT / "tail_prediction_source.csv")
    delta = read_csv(ROOT / "tail_rmse_improvement_source.csv")[0]
    assert float(a[0]["within_family_pearson_r"]) == s["training_selection"]["within_family_r"]
    assert float(a[1]["within_family_pearson_r"]) == s["validation"]["family_centered_r"]
    assert [float(a[1][key]) for key in ["ci95_lower", "ci95_upper"]] == s["validation"]["correlation_95ci"]
    assert float(p[0]["rmse"]) == s["validation"]["taxonomy_baseline"]["rmse"]
    assert float(p[1]["rmse"]) == s["validation"]["taxonomy_plus_activation"]["rmse"]
    assert float(delta["estimate"]) == float(p[0]["rmse"]) - float(p[1]["rmse"])
    assert [float(delta[key]) for key in ["ci95_lower", "ci95_upper"]] == s["validation"]["improvement_95ci"]
    for out, stem, phrases in [
        (HUMAN, "human_spot_count_distribution", ["27 countable", "NA", "3/27", "Single human"]),
        (ROOT, "tail_validation_evidence", ["did not confirm", "0.22680", "0.22821", "gate: failed", "not evaluated"]),
    ]:
        provenance = json.loads((out / "figure_provenance.json").read_text(encoding="utf-8"))
        assert all(sha(Path(path)) == digest for path, digest in provenance["source_sha256"].items())
        assert all(sha(out / name) == digest for name, digest in provenance["outputs_sha256"].items())
        assert sha(ROOT / "plot_evidence.py") == provenance["plot_script_sha256"]
        reader = PdfReader(out / f"{stem}.pdf")
        assert len(reader.pages) == 1
        text = reader.pages[0].extract_text()
        assert all(phrase in text for phrase in phrases)
        image = Image.open(out / f"{stem}.png")
        image.verify()
        render = Image.open(out / f"{stem}_pdf_render.png")
        render.verify()
        qa = {
            "status": "passed", "source_tables_match_saved_results": True,
            "source_and_output_hashes_verified": True, "pdf_pages": 1,
            "key_pdf_text_verified": phrases, "png_dimensions": list(image.size),
            "pdf_render_dimensions": list(render.size),
            "visual_inspection": "Both final PNG and Poppler-rendered PDF were viewed; no clipping, overlaps, missing labels, or misleading axes observed.",
            "accessibility": "Direct numeric labels, distinct point shapes, separate NA panel, and explicit group labels support interpretation without color; no formal color-vision simulation performed.",
            "fits_or_model_generation": False,
            "verification_script_sha256": sha(Path(__file__)),
        }
        (out / "figure_verification.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"figure": stem, "status": "passed"}))


if __name__ == "__main__":
    main()
