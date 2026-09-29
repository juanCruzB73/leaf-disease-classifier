"""Create a self-contained Colab bundle, without originals or old experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'colab_exports/gvlid_clean_colab.zip')
    args = parser.parse_args()
    paths = []
    for directory in ('data/gvlid_clean_processed', 'data/gvlid_clean_natural_eval'):
        paths.extend(p for p in (ROOT / directory).rglob('*') if p.is_file())
    for name in ('train.py', 'dataset.py', 'evaluate.py', 'run_gvlid_clean_experiment.py'):
        paths.append(ROOT / 'scripts' / name)
    for name in ('gvlid_integrity.json', 'gvlid_deduplication_report.csv', 'gvlid_split_manifest.csv'):
        paths.append(ROOT / 'reports/gvlid_clean_reviewed' / name)
    paths.extend(ROOT / p for p in ('reports/gvlid_decisions.csv', 'reports/gvlid_review_confirmation.json',
                                   'docs/gvlid_clean_colab.md', 'notebooks/gvlid_clean_colab.ipynb'))
    manifest = {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'files_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for path in sorted(paths):
            bundle.write(path, str(path.relative_to(ROOT)))
        bundle.writestr('colab_bundle_manifest.json', json.dumps(manifest, indent=2))
    print(f'{args.output}: {args.output.stat().st_size / 1024**2:.1f} MiB')


if __name__ == '__main__':
    main()
