import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_gvlid_clean_experiment as runner


class ExperimentTests(unittest.TestCase):
    def test_relocates_audit_without_using_old_dataset(self):
        row = dict(origin='natural', split='test', destination='/old/root/data/gvlid_clean_natural_eval/test/vl_black_rot/gvlid_000001.jpg')
        path = runner.destination(row)
        self.assertEqual(path, ROOT / 'data/gvlid_clean_natural_eval/test/vl_black_rot/gvlid_000001.jpg')
        row['destination'] = '/old/root/test/wrong_class/a.jpg'
        with self.assertRaises(ValueError):
            runner.destination(row)

    def test_training_command_fixed(self):
        cmd = runner.command(Path('/tmp/run'))
        for key, value in {'--epochs':'8','--finetune-epochs':'15','--lr':'0.001',
                           '--finetune-lr':'0.0001','--batch-size':'32','--seed':'42',
                           '--image-size':'224'}.items():
            self.assertEqual(cmd[cmd.index(key)+1], value)
        self.assertNotIn('--no-pretrained', cmd)
        self.assertNotIn('--limit-batches', cmd)
        self.assertIn(str(ROOT/'data/gvlid_clean_processed'), cmd)

    def test_notebook_code_compiles(self):
        nb=json.loads((ROOT/'notebooks/gvlid_clean_colab.ipynb').read_text())
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                source=''.join(line for line in cell['source'] if not line.startswith('%'))
                compile(source, '<notebook>', 'exec')

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'matplotlib installed in Colab')
    def test_exports_counts_and_normalized_confusion(self):
        import csv
        metrics=dict(per_class={c:{'precision':1.,'recall':1.,'f1-score':1.,'support':n}
                                for c,n in zip(runner.CLASSES,[164,106,14,23])},
                     accuracy=1.,macro_f1=1.,weighted_f1=1.,n_images=307,correct=307,incorrect=0,
                     confusion_matrix=[[164,0,0,0],[0,106,0,0],[0,0,14,0],[0,0,0,23]])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            runner.export_metrics(metrics,root,'test')
            with (root/'matriz_confusion_test.csv').open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(int(rows[2]['vl_black_rot']),14)
            self.assertTrue((root/'matriz_confusion_test_normalizada.png').exists())


if __name__ == '__main__':
    unittest.main()
