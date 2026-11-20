"""CLI wrapper for a new replication, preserving the recorded training script."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True,
                        help='New output directory. Existing directories are refused.')
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if out.exists():
        parser.error(f'Refusing existing output directory: {out}')
    import run_aligned_stability as experiment
    experiment.OUT = out
    experiment.main()


if __name__ == '__main__':
    main()
