import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from models.hl_decision.evaluation_checkpoints_test import dataset
from models.hl_decision.train_hl_decision_model import (
    AUXILIARY_TARGETS, HLMultihead, HLMultiheadNoAux, HLStateShapModel, TrainTorchModel,
)


class SelectiveAuxiliaryTest(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.object(TrainTorchModel, 'get_device', return_value=torch.device('cpu')))

    def test_selected_heads_only_require_selected_targets_and_report_selected_metrics(self):
        for heads in ([], ['heal'], ['pearl'], ['pot_soon'], ['pot_urgency'], ['heal', 'pearl']):
            with self.subTest(heads=heads):
                data = dataset()
                for name in ('pot_soon', 'pot_urgency'):
                    if name not in heads:
                        del data.Y[AUXILIARY_TARGETS[name]]
                model = HLMultihead(2, 4)
                trainer = TrainTorchModel(model=model)
                metrics = trainer.eval(dataloader=DataLoader(data, batch_size=2),
                                       auxiliary_heads=heads, inspect_initiation=True)
                self.assertEqual(set(metrics['components']), {'state', *heads})
                self.assertEqual(set(metrics['binary_metrics']), set(heads) - {'pot_urgency'})
                self.assertEqual(set(metrics['initiation']['auxiliary']), set(heads) & {'heal', 'pearl'})
                self.assertEqual(metrics['urgency_metrics'] is not None, 'pot_urgency' in heads)
                self.assertEqual(sum(map(sum, metrics['state_confusion_matrix'])), 6)

    def test_no_aux_train_reload_and_plot(self):
        data = dataset()
        del data.Y['target_pot_soon'], data.Y['target_pot_urgency']
        model = HLMultiheadNoAux(2, 4)
        trainer = TrainTorchModel(data, data, model=model, batch_size=3)
        with tempfile.TemporaryDirectory() as directory:
            result = trainer.train(save_path=Path(directory) / 'state.pt', epochs=1)
            for path in result['checkpoint_paths'].values():
                restored = trainer.load_model(path)
                self.assertIsInstance(restored, HLMultiheadNoAux)
                self.assertEqual(restored.checkpoint_meta['auxiliary_heads'], [])
                criterion = restored.checkpoint_meta['selection_criterion']
                expected_metric = 'avg_loss' if criterion == 'loss' else f'initiation.state.{criterion}.f1'
                self.assertEqual(restored.checkpoint_meta['selection_metric'], expected_metric)
            restored = trainer.load_model(result['checkpoint_paths']['loss'])
            torch.testing.assert_close(restored(data.X), model(data.X))
            torch.testing.assert_close(HLStateShapModel(restored)(data.X), restored(data.X))
            with patch('models.hl_decision.train_hl_decision_model.plt.show'):
                metrics = trainer.eval(restored, graph=True, inspect_initiation=True)
            self.assertEqual(set(metrics['components']), {'state'})

    def test_selection_validation_and_checkpoint_default(self):
        with self.assertRaises(ValueError):
            TrainTorchModel.resolve_auxiliary_heads(['unknown'])
        with self.assertRaises(ValueError):
            TrainTorchModel.resolve_auxiliary_heads(['heal'], HLMultiheadNoAux(2, 4))
        model = HLMultihead(2, 4)
        model.checkpoint_meta = {'auxiliary_heads': ['pearl']}
        self.assertEqual(TrainTorchModel.resolve_auxiliary_heads(model=model), ('pearl',))


if __name__ == '__main__':
    unittest.main()
