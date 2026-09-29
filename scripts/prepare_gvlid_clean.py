"""Audit original GVLiD globally, review near duplicates, then build clean splits.

Exit 2 means manual review is required (reports were successfully generated).
Each run requires a fresh report directory. No source files are modified.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import textwrap

import imagehash
from PIL import Image, ImageDraw

from prepare_gvlid_hybrid_data import GVLID_MAP, ROOT, SPLITS, image_files, split_images


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def phash(path):
    with Image.open(path) as im:
        return str(imagehash.phash(im.convert('RGB'), hash_size=8))


def write_csv(path, rows, fields):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def comparison(path, a, b, distance, source):
    canvas = Image.new('RGB', (1000, 680), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f'pHash distance: {distance}', fill='black')
    for x, row in ((0, a), (500, b)):
        label = row['original_class'] + '\n' + '\n'.join(textwrap.wrap(row['file'], 65))
        draw.multiline_text((x + 12, 35), label, fill='black')
        with Image.open(source / row['file']) as im:
            im = im.convert('RGB')
            im.thumbnail((490, 490))
            canvas.paste(im, (x + (500 - im.width) // 2, 180))
    canvas.save(path)


def audit(source, report, decisions=None):
    rows = []
    for original, target in sorted(GVLID_MAP.items()):
        for path in image_files(source / original):
            rows.append(dict(file=path.relative_to(source).as_posix(), original_class=original,
                             target_class=target, sha256=sha256(path), phash=phash(path)))
    uf = UnionFind(len(rows))
    for key in ('sha256', 'phash'):
        seen = {}
        for i, row in enumerate(rows):
            if row[key] in seen:
                uf.union(i, seen[row[key]])
            else:
                seen[row[key]] = i
    initial_groups = len({uf.find(i) for i in range(len(rows))})
    supplied = {}
    if decisions:
        with decisions.open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                key = row['pair_id']
                if key in supplied or row['decision'] not in ('duplicate', 'different', ''):
                    raise ValueError(f'Invalid/duplicate review decision: {key}')
                supplied[key] = row
    candidates = []
    for i, j in itertools.combinations(range(len(rows)), 2):
        distance = (int(rows[i]['phash'], 16) ^ int(rows[j]['phash'], 16)).bit_count()
        if not 1 <= distance <= 5:
            continue
        a, b = rows[i], rows[j]
        pair_id = hashlib.sha256(json.dumps([a['file'], a['sha256'], b['file'], b['sha256']]).encode()).hexdigest()
        decision = supplied.get(pair_id, {}).get('decision', '')
        candidates.append(dict(pair_id=pair_id, file_a=a['file'], file_b=b['file'],
                               class_a=a['target_class'], class_b=b['target_class'],
                               distance=distance, decision=decision, i=i, j=j,
                               status='near_duplicate_candidate'))
        if decision == 'duplicate':
            uf.union(i, j)
    if set(supplied) - {p['pair_id'] for p in candidates}:
        raise ValueError('Review file contains stale or unknown pairs; source content changed.')
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[uf.find(i)].append(i)
    kept, conflicts = [], []
    for root, members in sorted(groups.items()):
        conflict = len({rows[i]['target_class'] for i in members}) > 1
        if conflict:
            conflicts.append(members)
        seen_sha = set()
        for n, i in enumerate(members):
            row = rows[i]
            row['group_id'] = f'g{root:06d}'
            row['status'] = ('label_conflict' if conflict else 'kept' if n == 0 else
                             'exact_duplicate' if row['sha256'] in seen_sha else 'visual_duplicate')
            row['reason'] = ('Multiple labels: ' + ', '.join(sorted({rows[k]['target_class'] for k in members}))
                             if conflict else 'Representative' if n == 0 else 'SHA-256 / pHash=0 / reviewed duplicate component')
            seen_sha.add(row['sha256'])
        if not conflict:
            kept.append(members[0])
    pending = [p for p in candidates if not p['decision']]
    report.mkdir(parents=True, exist_ok=False)
    review = report / 'gvlid_near_duplicates_review'
    review.mkdir()
    for n, pair in enumerate(candidates):
        pair['preview'] = f'{n:06d}_{pair["pair_id"][:12]}.jpg'
        comparison(review / pair['preview'], rows[pair['i']], rows[pair['j']], pair['distance'], source)
    write_csv(report / 'gvlid_near_duplicate_candidates.csv', candidates,
              ['pair_id', 'file_a', 'file_b', 'class_a', 'class_b', 'distance', 'status', 'preview', 'decision'])
    write_csv(report / 'gvlid_deduplication_report.csv', rows,
              ['group_id', 'file', 'original_class', 'target_class', 'sha256', 'phash', 'status', 'reason'])
    summary = dict(original_images=len(rows), unique_sha256=len({r['sha256'] for r in rows}),
                   exact_duplicate_copies=len(rows) - len({r['sha256'] for r in rows}),
                   additional_phash_zero_merges=len({r['sha256'] for r in rows}) - initial_groups,
                   duplicate_groups=sum(len(g) > 1 for g in groups.values()),
                   label_conflict_groups=len(conflicts), conflict_images=sum(map(len, conflicts)),
                   retained_images=len(kept), retained_by_class=dict(sorted(Counter(rows[i]['target_class'] for i in kept).items())),
                   near_duplicate_pairs=len(candidates), pending_review_pairs=len(pending),
                   split_status='blocked_pending_review' if pending else 'ready',
                   phash_policy='64-bit pHash=0 grouped automatically; distances 1–5 require explicit review',
                   versions=dict(Pillow=Image.__version__, ImageHash=imagehash.__version__))
    (report / 'gvlid_deduplication_summary.txt').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return rows, kept, summary


def validate_outputs(records, controlled, hybrid, classes):
    # Rehash actual destination bytes, not just the proposed manifest.
    sets = {s: {k: set() for k in ('sha256', 'phash', 'group_id')} for s in SPLITS}
    for row in records:
        path = Path(row['destination'])
        digest, perceptual = sha256(path), phash(path)
        if digest != row['sha256'] or perceptual != row['phash']:
            raise ValueError(f'Copy integrity failed: {path}')
        if row.get('status') == 'label_conflict':
            raise ValueError(f'Label conflict included: {path}')
        for key, value in (('sha256', digest), ('phash', perceptual), ('group_id', row['group_id'])):
            sets[row['split']][key].add(value)
    for a, b in itertools.combinations(SPLITS, 2):
        for key in sets[a]:
            if sets[a][key] & sets[b][key]:
                raise ValueError(f'Leakage {key}: {a}/{b}')
    for split in SPLITS:
        for cls in classes:
            expected = Counter(sha256(p) for p in controlled[split, cls])
            actual = Counter(sha256(p) for p in (hybrid / split / cls).glob('controlled_*'))
            if actual != expected:
                raise ValueError(f'PlantVillage changed: {split}/{cls}')
    return {'all_split_sha256_intersections': 0, 'all_split_phash_zero_intersections': 0,
            'all_split_confirmed_group_intersections': 0, 'label_conflicts_included': 0,
            'controlled_all_splits_identical_to_reference': True}


def build(rows, kept, summary, args):
    classes = json.loads(args.source_label_map.read_text())
    if set(classes) != set(GVLID_MAP.values()) or set(classes.values()) != set(range(4)):
        raise ValueError('Invalid four-class label mapping')
    controlled = {}
    old_manifest = None
    if args.controlled_from_hybrid:
        old_manifest = json.loads((args.controlled_data / 'manifest.json').read_text())
        if old_manifest['label_map'] != classes:
            raise ValueError('Reference label maps differ')
    for split in SPLITS:
        for cls in classes:
            paths = image_files(args.controlled_data / split / cls)
            if args.controlled_from_hybrid:
                paths = [p for p in paths if p.name.startswith('controlled_')]
                if len(paths) != old_manifest['controlled_stats'][f'{split}/{cls}']:
                    raise ValueError(f'Controlled reference count mismatch: {split}/{cls}')
            if not paths:
                raise ValueError(f'Empty controlled reference: {split}/{cls}')
            controlled[split, cls] = paths
    records, counts = [], {}
    for offset, (original, cls) in enumerate(sorted(GVLID_MAP.items())):
        partitions = split_images([i for i in kept if rows[i]['target_class'] == cls], .15, .15, 42 + offset)
        counts[cls] = {s: len(partitions[s]) for s in SPLITS}
        for split, indices in partitions.items():
            directory = (args.hybrid_out if split == 'train' else args.natural_eval_out) / split / cls
            for i in indices:
                row = rows[i]
                records.append(dict(row, split=split, origin='natural',
                                    destination=str(directory / f'gvlid_{i:06d}{Path(row["file"]).suffix.lower()}')))
    # Create independent copies: no writable hard links back to originals.
    for output in (args.hybrid_out, args.natural_eval_out):
        output.mkdir(parents=True, exist_ok=False)
    for (split, cls), paths in controlled.items():
        directory = args.hybrid_out / split / cls
        directory.mkdir(parents=True)
        for n, path in enumerate(paths):
            target = directory / f'controlled_{n:06d}{path.suffix.lower()}'
            shutil.copy2(path, target)
            records.append(dict(file=str(path), sha256=sha256(path), phash=phash(path),
                                group_id='controlled_' + sha256(path), split=split, origin='controlled', destination=str(target)))
    for split in ('val', 'test'):
        for cls in classes:
            (args.natural_eval_out / split / cls).mkdir(parents=True)
    for row in records:
        if row['origin'] == 'natural':
            shutil.copy2(args.gvlid_data / row['file'], row['destination'])
    write_csv(args.report_dir / 'gvlid_split_manifest.csv', records,
              ['file', 'origin', 'target_class', 'group_id', 'split', 'sha256', 'phash', 'destination'])
    checks = validate_outputs(records, controlled, args.hybrid_out, classes)
    manifest = dict(summary, seed=42, natural_counts=counts, integrity=checks,
                    controlled_reference=str(args.controlled_data),
                    controlled_reference_is_previous_hybrid=args.controlled_from_hybrid,
                    label_map=classes, split_status='validated')
    for output in (args.hybrid_out, args.natural_eval_out):
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        (output / 'label_map.json').write_text(json.dumps(classes, indent=2))
    (args.report_dir / 'gvlid_integrity.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [('gvlid-data', ROOT / 'GVLiD'), ('report-dir', ROOT / 'reports'),
                          ('controlled-data', ROOT / 'data/controlled_processed'),
                          ('source-label-map', ROOT / 'models/resnet50/label_map.json'),
                          ('hybrid-out', ROOT / 'data/gvlid_clean_processed'),
                          ('natural-eval-out', ROOT / 'data/gvlid_clean_natural_eval')]:
        parser.add_argument('--' + name, type=Path, default=default)
    parser.add_argument('--decisions', type=Path, help='Reviewed candidate CSV: duplicate or different for every pair')
    parser.add_argument('--controlled-from-hybrid', action='store_true', help='Explicitly reuse controlled_* files from an earlier hybrid reference')
    args = parser.parse_args(argv)
    outputs = [args.report_dir.resolve(), args.hybrid_out.resolve(), args.natural_eval_out.resolve()]
    inputs = [args.gvlid_data.resolve(), args.controlled_data.resolve()]
    for output in outputs:
        if output.exists():
            raise FileExistsError(f'Output already exists; choose a fresh path: {output}')
        if any(output == p or output in p.parents or p in output.parents for p in inputs):
            raise ValueError(f'Output overlaps input: {output}')
    for a, b in itertools.combinations(outputs, 2):
        if a == b or a in b.parents or b in a.parents:
            raise ValueError('Output directories must not overlap')
    rows, kept, summary = audit(args.gvlid_data, args.report_dir, args.decisions)
    if summary['pending_review_pairs']:
        print('REVISIÓN MANUAL REQUERIDA: completar decision en el CSV (duplicate/different). No se crearon splits.', file=sys.stderr)
        return 2
    build(rows, kept, summary, args)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
