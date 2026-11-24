import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.hl_decision.train_hl_decision_model import HLMultihead, TrainTorchModel, test_model
from models.hl_decision.hl_decision_datasets import HLDecisionDataset
from models.torch_dataset import HLMultiTaskDataset


def dataset():
    # Consecutive positive start labels are deliberate: they must not be differenced.
    return HLMultiTaskDataset(np.zeros((6, 3, 2)), {
        "state": [0, 1, 1, 0, 1, 2],
        "heal_start": [0, 1, 1, 0, 1, 0],
        "pearl_start": [0, 0, 0, 0, 0, 1],
        "target_pot_soon": [0] * 6, "target_pot_urgency": [0] * 6,
        "round_id": [0, 0, 0, 1, 1, 1],
        "target_tick": [100, 101, 102, 100, 102, 103],
    })


class EvaluationCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.cpu = patch.object(TrainTorchModel, "get_device", return_value=torch.device("cpu"))
        self.cpu.start()
        self.addCleanup(self.cpu.stop)
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    def test_eval_uses_start_labels_and_shared_contiguous_endpoints(self):
        model = HLMultihead(2, 4)
        loader = DataLoader(dataset(), batch_size=2)
        result = TrainTorchModel(model=model).eval(dataloader=loader, inspect_initiation=True)
        # Only rows 1, 2, 5 have a preceding prediction from the same round/tick stream.
        main = result["initiation"]["state"]["heal"]
        auxiliary = result["initiation"]["auxiliary"]["heal"]
        self.assertEqual(main["positive_count"], 2)
        self.assertEqual(auxiliary["positive_count"], 2)
        self.assertEqual(main["sample_count"], 3)
        self.assertEqual(result["initiation"]["coverage"]["excluded_heal_starts"], 1)
        self.assertEqual(result["binary_metrics"]["heal"]["positive_count"], 3)
        for name in ("heal", "pearl"):
            self.assertEqual(result[f"{name}_initiation_f1"], result["initiation"]["state"][name]["f1"])

    def test_auxiliary_metrics_exclude_continuations_but_keep_other_negatives(self):
        data = dataset()
        data.Y["heal_start"][2] = 0  # A heal continuation with a confident prediction.
        model = HLMultihead(2, 4)
        with torch.no_grad():
            model.heal_head.weight.zero_()
            model.heal_head.bias.fill_(10.)
        result = TrainTorchModel(model=model).eval(
            dataloader=DataLoader(data, batch_size=2), inspect_initiation=True,
        )
        metrics = result["binary_metrics"]["heal"]
        self.assertEqual(metrics["sample_count"], 5)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 3)
        self.assertAlmostEqual(metrics["precision"], 2 / 5)
        self.assertEqual(metrics["recall"], 1.)
        self.assertAlmostEqual(metrics["f1"], 4 / 7)
        self.assertAlmostEqual(metrics["average_precision"], 2 / 5)
        self.assertEqual(result["initiation"]["auxiliary"]["heal"]["sample_count"], 2)
        self.assertEqual(result["initiation"]["state"]["heal"]["sample_count"], 3)
        self.assertEqual(np.asarray(result["state_confusion_matrix"]).sum(), 6)

    def test_checkpoint_criteria_are_independent_and_restore_loss_winner(self):
        model = HLMultihead(2, 4)
        trainer = TrainTorchModel(dataset(), dataset(), model=model, batch_size=6)
        # Auxiliary recall peaks at epoch 1; auxiliary F1 peaks at epochs 3 and 2.
        # Main-state F1 peaks at epoch 1, so it must not select these checkpoints.
        losses = [1., .5, .8]
        precision_recall = [((.1, .9), (.1, .9)),
                            ((.3, .3), (.8, .5)),
                            ((.8, .6), (.1, .7))]
        states = []

        def evaluate(model, **kwargs):
            states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
            epoch = len(states) - 1
            result = {"avg_loss": losses[epoch], "accuracy": .75, "initiation": {"state": {}, "auxiliary": {}}}
            for name, (precision, recall) in zip(("heal", "pearl"), precision_recall[epoch]):
                f1 = 2 * precision * recall / (precision + recall)
                result[f"{name}_initiation_f1"] = 1.0 if epoch == 0 else 0.0
                result["initiation"]["state"][name] = {"f1": result[f"{name}_initiation_f1"]}
                result["initiation"]["auxiliary"][name] = {"precision": precision, "recall": recall, "f1": f1}
            return result

        with tempfile.TemporaryDirectory() as directory, patch.object(trainer, "eval", side_effect=evaluate):
            path = Path(directory) / "run.pt"
            result = trainer.train("run.pt", save_path=path, epochs=3,
                                   checkpoint_meta={"features": ["a", "b"], "seq_len": 3})
            for suffix, epoch in [("loss", 2), ("heal", 3), ("pearl", 2)]:
                checkpoint = torch.load(path.with_name(f"run_{suffix}.pt"), weights_only=False)
                self.assertEqual(checkpoint["epoch"], epoch)
                self.assertEqual(checkpoint["validation_metrics"]["avg_loss"], losses[epoch - 1])
                metric = "avg_loss" if suffix == "loss" else f"initiation.auxiliary.{suffix}.f1"
                self.assertEqual(checkpoint["selection_metric"], metric)
                expected = checkpoint["validation_metrics"]["avg_loss"] if suffix == "loss" else checkpoint["validation_metrics"]["initiation"]["auxiliary"][suffix]["f1"]
                self.assertEqual(checkpoint["selection_score"], expected)
                self.assertIn("model_config", checkpoint)
                self.assertIn("training_config", checkpoint)
                for key, value in states[epoch - 1].items():
                    torch.testing.assert_close(checkpoint["model_state_dict"][key], value)
            self.assertEqual(result["best_epoch"], 2)
            for key, value in states[1].items():
                torch.testing.assert_close(model.state_dict()[key], value)

    def test_start_truth_is_not_differenced_and_predictions_never_bridge_gaps(self):
        identity = {k: dataset().Y[k].numpy() for k in ("round_id", "target_tick")}
        truth = {name: dataset().Y[f"{name}_start"].numpy() for name in ("heal", "pearl")}
        result = TrainTorchModel.score_initiation(
            np.array([0, 1, 1, 0, 1, 2]), truth, truth, identity,
        )
        self.assertEqual(result["state"]["heal"]["tp"], 1)
        self.assertEqual(result["state"]["heal"]["fn"], 1)
        self.assertEqual(result["auxiliary"]["heal"]["tp"], 2)
        self.assertEqual(result["state"]["pearl"]["tp"], 1)

    def test_sequence_metadata_matches_next_tick_labels_and_skips_gap_targets(self):
        ds = HLDecisionDataset.__new__(HLDecisionDataset)
        frames = pd.DataFrame({
            "round_id": [0] * 5 + [1] * 3,
            "recordingTickIndex": [0, 1, 2, 4, 5, 0, 1, 2],
            "label": ["attack", "heal", "heal", "attack", "pearl", "attack", "heal", "pearl"],
            "label_start": [0, 1, 2, 4, 5, 0, 1, 2],
            "myHp": np.arange(8), "target_pot_soon": [0] * 8,
            "target_pot_urgency": [0] * 8,
        })
        frames = ds._col_action_starts(frames)
        x, y = ds.select_seq_X_Y(frames, features=["myHp"], seq_len=1,
                                 target=ds.state_targets(frames), return_auxiliary_targets=True)
        np.testing.assert_array_equal(y["target_tick"].ravel(), [1, 2, 5, 1, 2])
        np.testing.assert_array_equal(y["round_id"].ravel(), [0, 0, 0, 1, 1])
        np.testing.assert_array_equal(y["heal_start"].ravel(), [1, 1, 0, 1, 0])
        np.testing.assert_array_equal(x[:, -1, 0], [0, 1, 3, 5, 6])

    def test_endpoint_identity_keeps_integer_precision(self):
        ds = HLMultiTaskDataset(np.zeros((2, 1, 1)), {"target_tick": [2**25, 2**25 + 1]})
        self.assertEqual(ds.Y["target_tick"].dtype, torch.int64)
        self.assertEqual(ds.Y["target_tick"][1] - ds.Y["target_tick"][0], 1)

    def test_architecture_and_legacy_checkpoints_reload_identically(self):
        x = torch.randn(2, 3, 2)
        for encoder in ("cnn", "linear", "lstm"):
            model = HLMultihead(2, 4, encoder_type=encoder).eval()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.pt"
                TrainTorchModel.save_model_checkpoint(model, path, {"features": ["a", "b"]})
                loaded = TrainTorchModel.load_model(path, device=torch.device("cpu"))
                legacy = HLMultihead.from_checkpoint({"model_state_dict": model.state_dict()}).eval()
                for key, expected in model(x).items():
                    torch.testing.assert_close(loaded(x)[key], expected)
                    torch.testing.assert_close(legacy(x)[key], expected)

    def test_test_entrypoint_passes_test_loader_and_checkpoint_sequence_length(self):
        model = HLMultihead(2, 4)
        model.checkpoint_meta = {"seq_len": 3, "features": ["a", "b"]}
        data = dataset()
        with patch("models.hl_decision.train_hl_decision_model.HLDecisionDataset") as factory, \
                patch.object(TrainTorchModel, "eval", return_value={}) as evaluate:
            factory.return_value.get_test_X_Y.return_value = (data.X, data.Y)
            test_model(hl_model=model)
            factory.return_value.get_test_X_Y.assert_called_once_with(seq_len=3, return_auxiliary_targets=True)
            loader = evaluate.call_args.kwargs["dataloader"]
            torch.testing.assert_close(loader.dataset.Y["state"], data.Y["state"])

    def test_zero_f1_checkpoints_and_patience(self):
        model = HLMultihead(2, 4)
        trainer = TrainTorchModel(dataset(), dataset(), model=model, batch_size=6)
        scores = [{"avg_loss": 1., "heal_initiation_f1": 0., "pearl_initiation_f1": 0.},
                  {"avg_loss": 1., "heal_initiation_f1": .2, "pearl_initiation_f1": 0.},
                  {"avg_loss": 1., "heal_initiation_f1": .2, "pearl_initiation_f1": 0.}]
        for score in scores:
            score["initiation"] = {"auxiliary": {name: {"f1": score[f"{name}_initiation_f1"]}
                                                 for name in ("heal", "pearl")}}
        with tempfile.TemporaryDirectory() as directory, patch.object(trainer, "eval", side_effect=scores):
            path = Path(directory) / "patience.pt"
            info = trainer.train(save_path=path, epochs=9, patience=1)
            self.assertEqual(info["epochs_run"], 3)
            self.assertEqual(info["best_epochs"], {"loss": 1, "heal": 2, "pearl": 1})
            self.assertTrue(path.with_name("patience_pearl.pt").exists())

    def test_saved_validation_metrics_reproduce_after_reload(self):
        trainer = TrainTorchModel(dataset(), dataset(), model=HLMultihead(2, 4), batch_size=2)
        with tempfile.TemporaryDirectory() as directory:
            info = trainer.train(save_path=Path(directory) / "actual.pt", epochs=1)
            path = info["checkpoint_paths"]["loss"]
            loaded = TrainTorchModel.load_model(path, device=torch.device("cpu"))
            actual = trainer.eval(loaded, inspect_initiation=True)
            expected = loaded.checkpoint_meta["validation_metrics"]
            self.assertAlmostEqual(actual["avg_loss"], expected["avg_loss"], places=7)
            self.assertEqual(actual["state_confusion_matrix"], expected["state_confusion_matrix"])
            self.assertEqual(actual["initiation"]["coverage"], expected["initiation"]["coverage"])
            for name in ("heal", "pearl"):
                for head in ("state", "auxiliary"):
                    for key in ("tp", "fp", "fn", "tn", "precision", "recall"):
                        self.assertEqual(actual["initiation"][head][name][key],
                                         expected["initiation"][head][name][key])


if __name__ == "__main__":
    unittest.main()
