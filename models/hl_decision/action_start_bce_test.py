import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.hl_decision.train_hl_decision_model import (
    HLActionStartModel, TrainTorchModel, HEAL_LOSS_WEIGHT, PEARL_LOSS_WEIGHT,
)
from models.torch_dataset import HLMultiTaskDataset


class ActionStartBceTest(unittest.TestCase):
    def setUp(self):
        self.data = HLMultiTaskDataset(torch.tensor([-.7, 1.2, -.2, .8]).reshape(4, 1, 1), {
            "state": [1, 2, 2, 0], "heal_start": [0, 0, 0, 0], "pearl_start": [0, 1, 0, 0],
        })
        self.cpu = patch.object(TrainTorchModel, "get_device", return_value=torch.device("cpu"))
        self.cpu.start()
        self.addCleanup(self.cpu.stop)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_single_logit_shape_and_action_specific_mask_weight(self):
        model = HLActionStartModel(input_dim=1, hidden_dim=4)
        self.assertEqual(model(self.data.X[:1]).shape, (1, 1))
        for action, index, weight in (("heal", 1, HEAL_LOSS_WEIGHT), ("pearl", 2, PEARL_LOSS_WEIGHT)):
            logits = model(self.data.X)
            truth = self.data.Y[f"{action}_start"]
            mask = (self.data.Y["state"] != index) | truth.bool()
            actual = TrainTorchModel.compute_start_losses(logits, self.data.Y, f"{action}_start")
            expected = nn.functional.binary_cross_entropy_with_logits(logits.ravel()[mask], truth[mask], pos_weight=weight)
            torch.testing.assert_close(actual, expected)
            grad, = torch.autograd.grad(actual, logits)
            self.assertEqual(grad.ravel()[~mask].abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            HLActionStartModel(output_dim=2)

    def test_eval_uses_sigmoid_and_eligible_weighted_mean_across_batches(self):
        model = HLActionStartModel(input_dim=1, hidden_dim=4)
        trainer = TrainTorchModel(model=model)
        logits = self.data.X[:, -1, 0]
        expected_loss = trainer.compute_start_losses(logits, self.data.Y, "pearl_start").item()
        with patch.object(model, "forward", side_effect=lambda x: x[:, -1, :1]), \
                patch.object(trainer, "print_binary_head_metrics") as metrics:
            for size in (1, 3, 4):
                loss = trainer.eval_action_start(action_name="pearl_start", dataloader=DataLoader(self.data, batch_size=size))
                self.assertAlmostEqual(loss, expected_loss, places=5)
                args = metrics.call_args.args
                torch.testing.assert_close(torch.as_tensor(args[2]), logits.sigmoid()[torch.tensor([True, True, False, True])])
                self.assertEqual(args[1].shape, (3,))
        torch.testing.assert_close(trainer.binary_logits_to_class(torch.tensor([-.1, .1])), torch.tensor([0, 1]))

    def test_pearl_training_routes_validation_and_reloads_checkpoint(self):
        model = HLActionStartModel(input_dim=1, hidden_dim=4)
        trainer = TrainTorchModel(self.data, self.data, model=model, batch_size=4)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(trainer, "eval_action_start", wraps=trainer.eval_action_start) as evaluate:
            path = Path(directory) / "pearl.pt"
            trainer.train_action_start(action_name="pearl_start", save_path=path, epochs=1)
            self.assertEqual(evaluate.call_args.kwargs["action_name"], "pearl_start")
            self.assertEqual(model.checkpoint_meta["action_name"], "pearl_start")
            restored = trainer.load_model(path)
            self.assertIsInstance(restored, HLActionStartModel)
            self.assertEqual(restored.checkpoint_meta["action_name"], "pearl_start")
            self.assertEqual(restored.checkpoint_meta["output_dim"], 1)
            torch.testing.assert_close(restored(self.data.X), model(self.data.X))


if __name__ == "__main__":
    unittest.main()
