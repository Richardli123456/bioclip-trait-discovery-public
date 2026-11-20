"""Describe the completed frozen-weight experiment; cosine scores are proxies."""
from pathlib import Path
import argparse
import csv
import json

import numpy as np
from PIL import Image

from experiment_common import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments/existing_sae_v1")
    args = parser.parse_args()
    out = args.out_dir
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    verification = json.loads((out / "verification.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed" and verification["status"] == "passed"
    records = manifest["images"]
    by_key = {(r["kind"], r["source"], r["condition"], r["seed"]): r for r in records}
    observations = []
    for name in protocol["intervention_species"]:
        conditions = list(dict.fromkeys(r["condition"] for r in records if r["kind"] == "intervention" and r["source"] == name))
        for condition in conditions:
            selected = [by_key["intervention", name, condition, seed] for seed in protocol["intervention_seeds"]]
            values = [r["image_bioclip_cosine_to_source"] for r in selected]
            deltas, pixel_deltas = [], []
            for record in selected:
                baseline = by_key["intervention", name, "original", record["seed"]]
                deltas.append(record["image_bioclip_cosine_to_source"] - baseline["image_bioclip_cosine_to_source"])
                with Image.open(out / record["image"]) as a, Image.open(out / baseline["image"]) as b:
                    pixel_deltas.append(float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).mean() / 255))
            observations.append({"species": name, "condition": condition, "seed_count": len(values),
                                 "mean_species_cosine": float(np.mean(values)),
                                 "std_species_cosine": float(np.std(values, ddof=1)) if len(values) > 1 else 0,
                                 "mean_cosine_change_from_original": float(np.mean(deltas)),
                                 "min_cosine_change_from_original": float(np.min(deltas)),
                                 "max_cosine_change_from_original": float(np.max(deltas)),
                                 "mean_rgb_difference_from_original": float(np.mean(pixel_deltas)),
                                 "input_change_norm": selected[0]["input_change_norm"]})
    with (out / "intervention_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(observations[0]))
        writer.writeheader()
        writer.writerows(observations)
    bridge_rows, bridge_summaries = [], []
    for bridge in protocol["bridges"]:
        source = bridge["source"]
        for seed in protocol["bridge_seeds"]:
            path = sorted((r for r in records if r["kind"] == "bridge" and r["source"] == source
                           and r["seed"] == seed and r["step"] is not None), key=lambda r: r["step"])
            scores = np.array([r["image_bioclip_cosine_to_target"] - r["image_bioclip_cosine_to_source"] for r in path])
            target_scores = np.array([r["image_bioclip_cosine_to_target"] for r in path])
            for record, difference in zip(path, scores):
                bridge_rows.append({"source": source, "target": bridge["target"], "seed": seed,
                                    "step": record["step"], "edited_feature": record["edited_feature"],
                                    "active_features": record["active_features"],
                                    "code_distance_to_target": record["code_distance_to_target"],
                                    "image_cosine_to_source": record["image_bioclip_cosine_to_source"],
                                    "image_cosine_to_target": record["image_bioclip_cosine_to_target"],
                                    "target_minus_source_cosine": float(difference), "image": record["image"]})
            bridge_summaries.append({"source": source, "target": bridge["target"], "seed": seed,
                                     "coordinate_edits": len(path) - 1,
                                     "target_minus_source_cosine_start": float(scores[0]),
                                     "target_minus_source_cosine_end": float(scores[-1]),
                                     "target_minus_source_cosine_gain": float(scores[-1] - scores[0]),
                                     "target_cosine_non_decreasing_step_fraction": float(np.mean(np.diff(target_scores) >= 0)),
                                     "relative_alignment_non_decreasing_step_fraction": float(np.mean(np.diff(scores) >= 0))})
    if bridge_rows:
        with (out / "bridge_scores.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(bridge_rows[0]))
            writer.writeheader()
            writer.writerows(bridge_rows)
    summary = {"training_performed": False, "image_count": len(records), "interventions": observations,
               "bridges": bridge_summaries,
               "score_interpretation": "Frozen BioCLIP image/text cosine is species alignment within the same model family, not a measured biological trait or independent identity benchmark.",
               "rgb_difference_interpretation": "Mean absolute RGB difference is sensitive to pose, background, framing and texture; it is not a trait score."}
    write_json(out / "summary.json", summary)
    # Standard static scientific plots, with every seed shown rather than significance claims.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conditions = [r["condition"] for r in observations if r["species"] == protocol["intervention_species"][0]]
    labels = ["original", "reconstruction", "feature 0x", "feature 0.5x", "feature 2x", "other feature 0x", "random direction"]
    fig, axes = plt.subplots(1, len(protocol["intervention_species"]), figsize=(15, 5), sharey=True, squeeze=False)
    for axis, name in zip(axes[0], protocol["intervention_species"]):
        for position, condition in enumerate(conditions):
            values = [by_key["intervention", name, condition, seed]["image_bioclip_cosine_to_source"] for seed in protocol["intervention_seeds"]]
            jitter = np.linspace(-0.12, 0.12, len(values))
            axis.scatter(position + jitter, values, s=28, color="#315b80", alpha=0.85)
            axis.plot([position - 0.2, position + 0.2], [np.mean(values)] * 2, color="#b34832", lw=2)
        axis.set_title(name, fontstyle="italic")
        axis.set_xticks(range(len(conditions)), labels, rotation=55, ha="right")
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("BioCLIP cosine to original species taxonomy")
    fig.suptitle("Frozen-weight interventions: dots are four image seeds; red lines are means", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "intervention_alignment.png", dpi=180)
    plt.close(fig)
    if bridge_summaries:
        fig, axes = plt.subplots(len(protocol["bridge_seeds"]), 1, figsize=(10, 3.5 * len(protocol["bridge_seeds"])), squeeze=False, sharex=True)
        for axis, seed in zip(axes[:, 0], protocol["bridge_seeds"]):
            path = [r for r in bridge_rows if r["seed"] == seed]
            axis.plot([r["step"] for r in path], [r["image_cosine_to_source"] for r in path], label="source species", color="#315b80")
            axis.plot([r["step"] for r in path], [r["image_cosine_to_target"] for r in path], label="target species", color="#b34832")
            axis.set_title(f"Seed {seed}")
            axis.set_ylabel("BioCLIP image/text cosine")
            axis.grid(alpha=0.2)
            axis.legend(loc="best")
        axes[-1, 0].set_xlabel("Number of SAE coordinates replaced with target values")
        fig.suptitle("Decoded owl bridge: species-alignment proxy across every generated step", fontsize=13)
        fig.tight_layout()
        fig.savefig(out / "bridge_alignment.png", dpi=180)
        plt.close(fig)
    print(json.dumps({"training_performed": False, "image_count": len(records), "bridges": bridge_summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
