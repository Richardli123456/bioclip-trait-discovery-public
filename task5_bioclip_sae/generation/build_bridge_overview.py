"""Audit completed bridge images and create a display-only PNG/PDF overview."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
except ImportError:  # PNG-only overview when reportlab is not installed
    canvas = None

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments/expanded_bridge_images_v2")
    args = parser.parse_args()
    out = args.out_dir
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    records = json.loads((out / "image_records.json").read_text(encoding="utf-8"))
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    count = protocol["image_count"]
    assert manifest["status"] == "completed" and manifest["image_count"] == count
    assert manifest["training_performed"] is False and manifest["model_updates"] is False
    assert len(records) == count
    assert len({record["image"] for record in records}) == count
    last = max(r["step"] for r in records if r["step"] is not None)
    shown = [round(last * k / 4) for k in (1, 2, 3)] + [last]
    originals = {}
    for record in records:
        path = out / record["image"]
        assert path.resolve().is_relative_to(out.resolve())
        assert sha(path) == record["sha256"]
        with Image.open(path) as image:
            assert image.size == (512, 512) and np.asarray(image).std() > 1
        originals[record["image"]] = record["sha256"]
    assert len(list((out / "images").glob("*.png"))) == count
    seeds = protocol["seeds"]
    assert len(seeds) == 2
    equal = {}
    for seed in seeds:
        source = next(r for r in records if r["seed"] == seed and r["condition"] == "source_original")
        step0 = next(r for r in records if r["seed"] == seed and r["step"] == 0)
        assert (out / source["image"]).read_bytes() == (out / step0["image"]).read_bytes()
        steps = sorted(r["step"] for r in records if r["seed"] == seed and r["step"] is not None)
        assert steps == list(range(last + 1))
        equal[str(seed)] = {"png_bytes_identical": True, "source": source["image"], "step0": step0["image"]}
    columns = [("source_original", None, ["Source original", "= bridge step 0"]),
               ("source_reconstruction", None, ["Source", "reconstruction"]),
               ("residual_preserving_bridge", shown[0], [f"Bridge step {shown[0]}"]),
               ("residual_preserving_bridge", shown[1], [f"Bridge step {shown[1]}"]),
               ("residual_preserving_bridge", shown[2], [f"Bridge step {shown[2]}"]),
               ("residual_preserving_bridge", last, [f"Bridge step {last}", "Target code +", "source residual"]),
               ("target_reconstruction", None, ["Target", "reconstruction"]),
               ("target_original", None, ["Target original"])]
    selected = [[next(r for r in records if r["seed"] == seed and r["condition"] == condition and r["step"] == step)
                 for condition, step, _ in columns] for seed in seeds]
    cell, gap, margin = 256, 12, 40
    width, height = 2 * margin + 8 * cell + 7 * gap, 962
    sheet = Image.new("RGB", (width, height), "#f5f7fa")
    draw = ImageDraw.Draw(sheet)
    pdf_path, png_path = out / "bridge_overview.pdf", out / "bridge_overview.png"
    assert not pdf_path.exists() and not png_path.exists(), "Refusing overview overwrite"
    pdf = canvas.Canvas(str(pdf_path), pagesize=(width / 2, height / 2)) if canvas else None
    if pdf:
        pdf.setTitle(f"Exploratory SAE species bridge: {protocol['source']} to {protocol['target']}")
        pdf.setFillColorRGB(245 / 255, 247 / 255, 250 / 255)
        pdf.rect(0, 0, width / 2, height / 2, fill=1, stroke=0)

    def text(x, y, value, size=22, bold=False, color="#173648"):
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf", size)
        draw.text((x, y), value, font=font, fill=color)
        if pdf:
            pdf.setFont("Helvetica-Bold" if bold else "Helvetica", size / 2)
            pdf.setFillColor(color)
            pdf.drawString(x / 2, (height - y - size) / 2 + 2, value)

    text(margin, 22, "Exploratory SAE species bridge", 36, True)
    text(margin, 70, f"{protocol['source']}  ->  {protocol['target']}", 29)
    text(margin, 112, f"{last} one-coordinate updates; original source residual retained throughout. Fixed seed within each row.", 23)
    for col, (_, _, lines) in enumerate(columns):
        left = margin + col * (cell + gap)
        for line, label in enumerate(lines):
            text(left, 160 + line * 25, label, 21, line == 0)
    for row, seed in enumerate(seeds):
        y = 277 + row * 310
        text(margin, y - 37, f"Seed {seed}", 24, True)
        for col, record in enumerate(selected[row]):
            x = margin + col * (cell + gap)
            with Image.open(out / record["image"]) as image:
                sheet.paste(image.convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS), (x, y))
                # Embed original 512-pixel images in the PDF at reduced display size.
                if pdf:
                    pdf.drawImage(ImageReader(image), x / 2, (height - y - cell) / 2, width=cell / 2, height=cell / 2)
    text(margin, 867, "Display note: PNG thumbnails reduce 512 x 512 originals to 256 x 256." + (" PDF embeds original pixels at reduced display size." if pdf else ""), 20)
    text(margin, 898, f"All {last + 1} bridge states (steps 0-{last}) remain as full-resolution files. Intermediate states are omitted only from this overview.", 20)
    text(margin, 929, "Exploratory representation path only: no trait score or causal trait claim is made here.", 20)
    sheet.save(png_path)
    if pdf:
        pdf.showPage()
        pdf.save()
    assert all(sha(out / filename) == checksum for filename, checksum in originals.items())
    audit = {"status": "passed", "execution": "CPU-only saved-image audit and layout; no model inference or training",
             "image_count": len(records), "unique_filenames": len(originals), "all_image_hashes_passed": True,
             "all_images_512_square": True, "steps_per_seed": list(range(last + 1)), "source_original_equals_step0": equal,
             "original_images_unchanged_after_layout": True, "overview_columns": [c[2] for c in columns],
             "selected_records": selected, "png_display_resolution_per_cell": [256, 256],
             "original_resolution_per_image": [512, 512], "pdf_written": bool(pdf),
             "pdf_preserves_original_embedded_pixels": True if pdf else None,
             "visual_layout_review": "pending", "spot_scoring_performed": False,
             "source_manifest_sha256": sha(out / "run_manifest.json"), "builder_sha256": sha(Path(__file__)),
             "overview_sha256": {p.name: sha(p) for p in (png_path, pdf_path) if p.exists()}}
    (out / "bridge_overview_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"overview_created": True, "audited_images": count, "source_step0_png_identity": equal}, indent=2), flush=True)


if __name__ == "__main__":
    main()
