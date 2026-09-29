import argparse
import json
import random
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_gvlid_clean as clean
from PIL import Image


class CleanTests(unittest.TestCase):
    def test_transitive_union(self):
        uf = clean.UnionFind(4)
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(1, 3)
        self.assertEqual(len({uf.find(i) for i in range(4)}), 1)

    def test_conflict_and_manual_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            for n, cls in enumerate(sorted(clean.GVLID_MAP)):
                directory = source / cls
                directory.mkdir(parents=True)
                Image.new('RGB', (10, 10), (n * 50, 30, 80)).save(directory / 'a.png')
            # Same perceptual content with conflicting labels, plus near pair.
            hashes = ['0000000000000000', '0000000000000000', '0000000000000003', 'ffffffffffffffff']
            with patch.object(clean, 'phash', side_effect=hashes), contextlib.redirect_stdout(io.StringIO()):
                rows, kept, summary = clean.audit(source, root / 'report')
            self.assertEqual(summary['label_conflict_groups'], 1)
            self.assertEqual(summary['conflict_images'], 2)
            self.assertEqual(summary['pending_review_pairs'], 2)
            self.assertEqual(len(kept), 2)
            self.assertEqual(len(list((root / 'report/gvlid_near_duplicates_review').glob('*.jpg'))), 2)
            self.assertTrue(all(rows[i]['status'] == 'kept' for i in kept))

    def test_leakage_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'a.png'
            Image.new('RGB', (10, 10)).save(path)
            row = dict(destination=str(path), sha256=clean.sha256(path), phash=clean.phash(path), group_id='g1')
            with self.assertRaisesRegex(ValueError, 'Leakage'):
                clean.validate_outputs([dict(row, split='train'), dict(row, split='test')], {}, Path(tmp), [])

    def test_complete_build_and_baseline_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, controlled = root / 'source', root / 'controlled'
            rng = random.Random(42)
            def make_image(path):
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.frombytes('RGB', (32, 32), rng.randbytes(32 * 32 * 3)).save(path)
            rows = []
            for original, cls in sorted(clean.GVLID_MAP.items()):
                for n in range(10):
                    path = source / original / f'{n}.png'
                    make_image(path)
                    rows.append(dict(file=path.relative_to(source).as_posix(), target_class=cls,
                                     sha256=clean.sha256(path), phash=clean.phash(path),
                                     group_id=f'g{len(rows)}', status='kept'))
                for split in clean.SPLITS:
                    make_image(controlled / split / cls / 'a.png')
            labels = root / 'labels.json'
            labels.write_text(json.dumps({c: n for n, c in enumerate(sorted(clean.GVLID_MAP.values()))}))
            report = root / 'report'
            report.mkdir()
            args = argparse.Namespace(source_label_map=labels, controlled_from_hybrid=False,
                                      controlled_data=controlled, gvlid_data=source,
                                      hybrid_out=root / 'hybrid', natural_eval_out=root / 'eval', report_dir=report)
            with contextlib.redirect_stdout(io.StringIO()):
                clean.build(rows, list(range(len(rows))), {}, args)
            manifest = json.loads((args.hybrid_out / 'manifest.json').read_text())
            self.assertEqual(manifest['split_status'], 'validated')
            self.assertEqual(sum(c['train'] for c in manifest['natural_counts'].values()), 24)
            self.assertTrue(manifest['integrity']['controlled_all_splits_identical_to_reference'])

    def test_split_deterministic(self):
        first = clean.split_images(list(range(100)), .15, .15, 42)
        self.assertEqual(first, clean.split_images(list(range(100)), .15, .15, 42))
        self.assertEqual({k: len(v) for k, v in first.items()}, dict(train=70, val=15, test=15))
        self.assertEqual(len(set(sum(first.values(), []))), 100)


if __name__ == '__main__':
    unittest.main()
