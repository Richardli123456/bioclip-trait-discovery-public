"""Rank every recorded species by binary SAE-support overlap, without training."""
from pathlib import Path
import argparse
import csv
import time

import numpy as np

from experiment_common import ALIGNED, ROOT, EPS, FrozenArtifacts, rank_neighbors, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-dir", type=Path, default=ALIGNED)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "overlap")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    started = time.perf_counter()
    data = FrozenArtifacts(args.aligned_dir)
    if not 1 <= args.top_k < len(data.rows):
        raise ValueError("top-k must be between 1 and species_count - 1")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    support = data.z > EPS
    sizes = support.sum(1)
    postings = [np.flatnonzero(support[:, j]) for j in range(support.shape[1])]
    rng = np.random.default_rng(20261002)
    null_peers = (np.arange(len(data.rows)) + rng.integers(1, len(data.rows), len(data.rows))) % len(data.rows)
    top_shared, null_shared, same_genus, same_family, ties = [], [], [], [], []
    fields = ["source", "neighbor", "rank", "shared_active", "source_active", "neighbor_active",
              "jaccard", "same_genus", "same_family", "source_split", "neighbor_split",
              "top_overlap_tie_count"]
    with (args.out_dir / "neighbors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(data.rows):
            # Inverted lists avoid a dense 5,388 x 5,388 x 2,048 comparison and new dependencies.
            ids = np.concatenate([postings[j] for j in np.flatnonzero(support[i])])
            counts = np.bincount(ids, minlength=len(data.rows))
            ranked = rank_neighbors(counts, data.names, i, args.top_k)
            maximum = int(counts[ranked[0]])
            tie_count = int(np.count_nonzero(counts[np.arange(len(counts)) != i] == maximum))
            top_shared.append(maximum)
            null_shared.append(int(counts[null_peers[i]]))
            same_genus.append(row["genus"] == data.rows[ranked[0]]["genus"])
            same_family.append(row["family"] == data.rows[ranked[0]]["family"])
            ties.append(tie_count)
            for rank, j in enumerate(ranked, start=1):
                peer = data.rows[j]
                writer.writerow({"source": row["scientific_name"], "neighbor": peer["scientific_name"],
                                 "rank": rank, "shared_active": int(counts[j]),
                                 "source_active": int(sizes[i]), "neighbor_active": int(sizes[j]),
                                 "jaccard": float(counts[j] / (sizes[i] + sizes[j] - counts[j])),
                                 "same_genus": row["genus"] == peer["genus"],
                                 "same_family": row["family"] == peer["family"],
                                 "source_split": row["split"], "neighbor_split": peer["split"],
                                 "top_overlap_tie_count": tie_count})
    summary = {"training_performed": False, "scope": "Existing 5,388-species aligned SAE",
               "threshold": EPS, "ranking": "Descending shared active features; alphabetical scientific-name ties",
               "jaccard": "intersection / union; all unedited codes have 32 active features",
               "top_k": args.top_k, "row_count": len(data.rows) * args.top_k,
               "mean_top1_shared": float(np.mean(top_shared)),
               "median_top1_shared": float(np.median(top_shared)),
               "mean_random_pair_shared": float(np.mean(null_shared)),
               "random_pair_rule": "One uniformly sampled non-self peer per species; seed 20261002",
               "top1_same_genus_fraction": float(np.mean(same_genus)),
               "top1_same_family_fraction": float(np.mean(same_family)),
               "top1_tied_fraction": float(np.mean(np.asarray(ties) > 1)),
               "provenance": data.provenance(), "elapsed_seconds": time.perf_counter() - started,
               "limits": ["Overlap is a representation similarity, not a biological trait measurement",
                          "All splits are searched together; this is descriptive retrieval, not held-out validation",
                          "Feature identity is specific to the saved aligned checkpoint"]}
    write_json(args.out_dir / "summary.json", summary)
    print({k: v for k, v in summary.items() if k != "provenance"}, flush=True)


if __name__ == "__main__":
    main()
