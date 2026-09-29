"""Fixed clean experiment runner for Colab. Reuses train.py and evaluate.py.

preflight: validate existing data without modifying it.
train: preflight, then the original 8+15 epoch pipeline.
evaluate: completed run only; export metrics and confusion matrices.
prepare-baseline: copy only controlled files to a separate baseline dataset.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys

import imagehash
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['healthy_leaf', 'vl_black_measles', 'vl_black_rot', 'vl_leaf_blight']
LABELS = dict(zip(CLASSES, range(4)))
NATURAL = {'train': [763, 498, 68, 105], 'val': [164, 106, 14, 23], 'test': [164, 106, 14, 23]}
CONTROLLED = {'train': [339, 1107, 944, 860], 'val': [42, 138, 118, 108], 'test': [42, 138, 118, 108]}
CONFIG = dict(seed=42, architecture='resnet50', initial_weights='IMAGENET1K_V2',
              epochs=8, finetune_epochs=15, lr=.001, finetune_lr=.0001,
              batch_size=32, image_size=224, optimizer='Adam',
              loss='CrossEntropyLoss: total / (4 * class_count)',
              checkpoint_selection='maximum macro F1 on controlled PlantVillage validation',
              transforms=dict(train=['RandomResizedCrop(224, scale=(0.8,1.0))', 'RandomHorizontalFlip()',
                                     'RandomRotation(15)', 'ColorJitter(brightness=.2,contrast=.2,saturation=.2)',
                                     'ToTensor', 'ImageNet Normalize'],
                              evaluation=['Resize((224,224))', 'ToTensor', 'ImageNet Normalize']))


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csv_write(path, rows, fields):
    with path.open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def destination(row):
    # Audit has absolute paths from the original workstation. Relocate by known
    # dataset + split/class/filename, never by arbitrary basename search.
    base = 'gvlid_clean_processed' if row['origin'] == 'controlled' or row['split'] == 'train' else 'gvlid_clean_natural_eval'
    tail = Path(row['destination']).parts[-3:]
    if tail[0] != row['split'] or tail[1] not in CLASSES:
        raise ValueError('Invalid audit destination')
    return ROOT / 'data' / base / Path(*tail)


def preflight():
    report = ROOT / 'reports/gvlid_clean_reviewed'
    integrity = read(report / 'gvlid_integrity.json')
    if integrity['split_status'] != 'validated' or integrity['pending_review_pairs'] != 0:
        raise ValueError('Audit is not validated or has pending decisions')
    with (ROOT / 'reports/gvlid_decisions.csv').open(newline='') as f:
        decisions = list(csv.DictReader(f))
    if len(decisions) != 17 or any(r['decision'] != 'duplicate' for r in decisions):
        raise ValueError('Expected 17 confirmed duplicate decisions')
    with (report / 'gvlid_deduplication_report.csv').open(newline='') as f:
        audited = {r['file']: r for r in csv.DictReader(f)}
    with (report / 'gvlid_split_manifest.csv').open(newline='') as f:
        records = list(csv.DictReader(f))
    expected_paths = set()
    counts = Counter()
    sets = {s: {k: set() for k in ('sha256', 'phash', 'group_id')} for s in NATURAL}
    for row in records:
        path = destination(row)
        if path in expected_paths:
            raise ValueError(f'Repeated destination: {path}')
        expected_paths.add(path)
        if digest(path) != row['sha256']:
            raise ValueError(f'SHA mismatch: {path}')
        with Image.open(path) as im:
            phash = str(imagehash.phash(im.convert('RGB'), hash_size=8))
        if phash != row['phash']:
            raise ValueError(f'pHash mismatch: {path}')
        if row['origin'] == 'natural':
            original = audited[row['file']]
            if any(original[k] != row[k] for k in ('group_id', 'sha256', 'phash', 'target_class')) or original['status'] != 'kept':
                raise ValueError(f'Excluded or modified natural entry: {path}')
        elif row['origin'] != 'controlled':
            raise ValueError('Unknown origin')
        counts[row['origin'], row['split'], path.parent.name] += 1
        for key in sets[row['split']]:
            sets[row['split']][key].add(row[key])
    for a, b in itertools.combinations(NATURAL, 2):
        for key in sets[a]:
            if sets[a][key] & sets[b][key]:
                raise ValueError(f'Leakage {key}: {a}/{b}')
    actual_paths = set()
    for name, splits in [('gvlid_clean_processed', NATURAL), ('gvlid_clean_natural_eval', ('val', 'test'))]:
        base = ROOT / 'data' / name
        if read(base / 'label_map.json') != LABELS:
            raise ValueError(f'Incorrect labels: {base}')
        for split in splits:
            if {p.name for p in (base / split).iterdir() if p.is_dir()} != set(CLASSES):
                raise ValueError(f'Incorrect classes: {base}/{split}')
            actual_paths.update(p for p in (base / split).rglob('*') if p.is_file())
    if actual_paths != expected_paths:
        raise ValueError('Dataset inventory differs from audited inventory')
    for origin, expected in [('natural', NATURAL), ('controlled', CONTROLLED)]:
        for split, values in expected.items():
            for cls, n in zip(CLASSES, values):
                if counts[origin, split, cls] != n:
                    raise ValueError(f'Count mismatch: {origin}/{split}/{cls}')
    return dict(status='passed', natural=NATURAL, controlled=CONTROLLED,
                hybrid_train=4684, controlled_test=406, natural_test=307,
                sha256_intersections=0, phash_zero_intersections=0, confirmed_group_intersections=0,
                audit_sha256=digest(report / 'gvlid_split_manifest.csv'),
                integrity_sha256=digest(report / 'gvlid_integrity.json'),
                controlled_reference='Previous hybrid controlled subset; independent original baseline provenance unavailable')


def command(run, baseline=False):
    data = ROOT / 'data' / ('gvlid_clean_controlled' if baseline else 'gvlid_clean_processed')
    return [sys.executable, '-u', str(ROOT / 'scripts/train.py'), '--model', 'resnet50',
            '--data-dir', str(data), '--label-map', str(ROOT / 'data/gvlid_clean_processed/label_map.json'),
            '--models-dir', str(run / 'models'), '--run-name', 'baseline' if baseline else 'hybrid',
            '--epochs', '8', '--finetune-epochs', '15', '--lr', '0.001', '--finetune-lr', '0.0001',
            '--batch-size', '32', '--image-size', '224', '--seed', '42', '--num-workers', '0']


def execute(cmd, log):
    with log.open('x') as f:
        process = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                f.write(line)
                f.flush()
            if process.wait():
                raise RuntimeError(f'Command failed; see {log}')
        except BaseException:
            process.terminate()
            process.wait()
            raise


def train(run, baseline):
    checks = preflight()
    import torch
    import torchvision
    import PIL
    import sklearn
    if not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime in Colab before training; no CPU fallback.')
    modeldir = run / 'models' / ('baseline' if baseline else 'hybrid')
    if modeldir.exists():
        raise FileExistsError(f'Run already exists: {modeldir}; choose a fresh run directory')
    if baseline:
        verify_baseline_data()
    run.mkdir(parents=True, exist_ok=True)
    manifest_path = run / ('baseline_manifest.json' if baseline else 'experiment_manifest.json')
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = read(ROOT / 'colab_bundle_manifest.json')['git_commit'] if (ROOT / 'colab_bundle_manifest.json').exists() else None
    cmd = command(run, baseline)
    manifest = dict(CONFIG, started_at=datetime.now(timezone.utc).isoformat(), status='training',
                    classes=LABELS, preflight=checks, git_commit=commit, command=cmd,
                    data_paths={'hybrid': str(ROOT / 'data/gvlid_clean_processed'), 'natural': str(ROOT / 'data/gvlid_clean_natural_eval')},
                    training_counts={'PlantVillage': 3250, 'GVLiD': 0 if baseline else 1434},
                    gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                    versions={'torchvision': torchvision.__version__, 'Pillow': PIL.__version__,
                              'ImageHash': imagehash.__version__, 'sklearn': sklearn.__version__,
                              'python': sys.version},
                    source_hashes={f: digest(ROOT / 'scripts' / f) for f in ('train.py', 'dataset.py', 'evaluate.py', 'run_gvlid_clean_experiment.py')},
                    baseline_status='Original checkpoint exists but no split/checkpoint provenance; comparison unavailable until comparable baseline retrained')
    save(manifest_path, manifest)
    try:
        execute(cmd, run / ('baseline_training.log' if baseline else 'hybrid_training.log'))
        history = read(modeldir / 'train_history.json')
        if len(history['history']) != 23:
            raise ValueError('Incomplete training history')
        best = max(history['history'], key=lambda r: r['val_macro_f1'])
        initial = Path(torch.hub.get_dir()) / 'checkpoints/resnet50-11ad3fa6.pth'
        manifest['initial_weights_sha256'] = digest(initial)
        manifest.update(status='trained', completed_at=datetime.now(timezone.utc).isoformat(),
                        checkpoint=str(modeldir / 'best_model.pt'), checkpoint_sha256=digest(modeldir / 'best_model.pt'),
                        best_validation=best)
        save(manifest_path, manifest)
    except BaseException as error:
        manifest.update(status='failed', error=str(error))
        save(manifest_path, manifest)
        raise


def verify_baseline_data():
    for split in CONTROLLED:
        for cls in CLASSES:
            source = ROOT / 'data/gvlid_clean_processed' / split / cls
            target = ROOT / 'data/gvlid_clean_controlled' / split / cls
            expected = {p.name: digest(p) for p in source.glob('controlled_*')}
            actual = {p.name: digest(p) for p in target.iterdir() if p.is_file()}
            if expected != actual:
                raise ValueError(f'Baseline data mismatch: {split}/{cls}')


def prepare_baseline():
    preflight()
    target = ROOT / 'data/gvlid_clean_controlled'
    target.mkdir(exist_ok=False)
    for split in CONTROLLED:
        for cls in CLASSES:
            output = target / split / cls
            output.mkdir(parents=True)
            for path in (ROOT / 'data/gvlid_clean_processed' / split / cls).glob('controlled_*'):
                shutil.copy2(path, output / path.name)
    verify_baseline_data()


def export_metrics(metrics, directory, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    per_class = [dict(clase=c, **metrics['per_class'][c]) for c in CLASSES]
    for key in ('macro_f1', 'weighted_f1', 'accuracy', 'n_images', 'correct', 'incorrect'):
        for row in per_class:
            row[key] = metrics[key]
    csv_write(directory / f'metricas_{name}.csv', per_class, list(per_class[0]))
    cm = np.array(metrics['confusion_matrix'])
    for normalized in (False, True):
        matrix = cm / cm.sum(axis=1, keepdims=True) if normalized else cm
        suffix = '_normalizada' if normalized else ''
        rows = [dict(clase_real=cls, **dict(zip(CLASSES, values.tolist()))) for cls, values in zip(CLASSES, matrix)]
        csv_write(directory / f'matriz_confusion_{name}{suffix}.csv', rows, ['clase_real'] + CLASSES)
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=1 if normalized else None)
        ax.set(xticks=range(4), yticks=range(4), xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel='Predicción', ylabel='Clase real', title=name)
        plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
        for i, j in itertools.product(range(4), repeat=2):
            ax.text(j, i, f'{matrix[i,j]:.3f}' if normalized else str(matrix[i,j]), ha='center', color='black')
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(directory / f'matriz_confusion_{name}{suffix}.png', dpi=160)
        plt.close(fig)


def evaluate(run):
    checks = preflight()
    manifest = read(run / 'experiment_manifest.json')
    if manifest['status'] != 'trained' or manifest['preflight']['audit_sha256'] != checks['audit_sha256']:
        raise ValueError('A completed training run on this exact audit is required')
    out = run / 'evaluation'
    out.mkdir(exist_ok=False)
    jobs = [('hybrid', 'gvlid_clean_processed', 'hibrido_controlado'),
            ('hybrid', 'gvlid_clean_natural_eval', 'hibrido_natural')]
    if (run / 'baseline_manifest.json').exists():
        bm = read(run / 'baseline_manifest.json')
        if bm['status'] != 'trained' or bm['preflight']['audit_sha256'] != checks['audit_sha256']:
            raise ValueError('Baseline run not verified')
        jobs.append(('baseline', 'gvlid_clean_natural_eval', 'baseline_natural'))
    results = {}
    for model, data, name in jobs:
        provenance = read(run / ('baseline_manifest.json' if model == 'baseline' else 'experiment_manifest.json'))
        checkpoint = run / 'models' / model / 'best_model.pt'
        if digest(checkpoint) != provenance['checkpoint_sha256']:
            raise ValueError(f'Checkpoint changed: {checkpoint}')
        cmd = [sys.executable, '-u', str(ROOT / 'scripts/evaluate.py'), '--model', 'resnet50',
               '--data-dir', str(ROOT / 'data' / data), '--models-dir', str(run / 'models'), '--run-name', model,
               '--label-map', str(ROOT / 'data/gvlid_clean_processed/label_map.json'),
               '--image-size', '224', '--batch-size', '32', '--output', str(out / f'{name}.json')]
        execute(cmd, out / f'{name}.log')
        result = read(out / f'{name}.json')
        if result['n_images'] != (406 if data == 'gvlid_clean_processed' else 307):
            raise ValueError('Wrong evaluation count')
        export_metrics(result, out, name)
        results[name] = result
    if 'baseline_natural' in results:
        b, h = results['baseline_natural'], results['hibrido_natural']
        if [(p['file'], p['true']) for p in b['predictions']] != [(p['file'], p['true']) for p in h['predictions']]:
            raise ValueError('Different natural tests')
        rows = [dict(modelo=name, **{k: r[k] for k in ('accuracy', 'macro_f1', 'weighted_f1', 'correct', 'incorrect')})
                for name, r in [('PlantVillage baseline', b), ('Híbrido limpio', h)]]
        csv_write(out / 'comparacion_global_natural.csv', rows, list(rows[0]))
        rows = []
        for cls in CLASSES:
            row = dict(clase=cls, support=b['per_class'][cls]['support'])
            for model, result in [('baseline', b), ('hibrido', h)]:
                for metric in ('precision', 'recall', 'f1-score'):
                    row[f'{metric}_{model}'] = result['per_class'][cls][metric]
            rows.append(row)
        csv_write(out / 'comparacion_por_clase_natural.csv', rows, list(rows[0]))
    else:
        (out / 'BASELINE_PENDIENTE.txt').write_text('No hay baseline con procedencia verificada. No se genera comparación final. Preparar y entrenar baseline con el notebook.\n')
    save(out / 'evaluation_manifest.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
         baseline_available='baseline_natural' in results, preflight=checks,
         limitation='vl_black_rot has only 14 natural test images; no test-driven tuning permitted'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'train', 'prepare-baseline', 'evaluate'])
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'resultados_comparacion_clean')
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if args.action == 'preflight':
        print(json.dumps(preflight(), indent=2))
    elif args.action == 'train':
        train(run, args.baseline)
    elif args.action == 'prepare-baseline':
        prepare_baseline()
    else:
        evaluate(run)


if __name__ == '__main__':
    main()
