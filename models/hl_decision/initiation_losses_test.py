import unittest

import torch
from torch import nn

from models.hl_decision.train_hl_decision_model import TrainTorchModel, CLASS_WEIGHTS, LOSS_WEIGHTS


class InitiationLossTest(unittest.TestCase):
    def inputs(self, states, heal_starts, pearl_starts):
        count = len(states)
        targets = {
            "state": torch.tensor(states).reshape(-1, 1),
            "heal_start": torch.tensor(heal_starts).reshape(-1, 1),
            "pearl_start": torch.tensor(pearl_starts).reshape(-1, 1),
            "target_pot_soon": torch.zeros(count, 1),
            "target_pot_urgency": torch.zeros(count, 1),
        }
        pred = {
            "state_logits": torch.zeros(count, 6, requires_grad=True),
            "heal_logits": torch.linspace(-1, 2, count, requires_grad=True),
            "pearl_logits": torch.linspace(2, -1, count, requires_grad=True),
            "pot_soon_logits": torch.zeros(count, requires_grad=True),
            "pot_urgency": torch.full((count,), .5, requires_grad=True),
        }
        return pred, targets

    def losses(self, pred, targets):
        return TrainTorchModel.compute_losses(
            pred, targets, nn.CrossEntropyLoss(weight=CLASS_WEIGHTS, reduction="none"),
            nn.BCEWithLogitsLoss(), nn.SmoothL1Loss(),
            heal_criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.)),
            pearl_criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.)),
        )

    def test_masks_use_next_state_and_keep_starts_and_other_action_negatives(self):
        pred, targets = self.inputs([0, 1, 1, 2, 2, 4], [0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0])
        total, losses, state = self.losses(pred, targets)
        masks = {"heal": torch.tensor([True, True, False, True, True, True]),
                 "pearl": torch.tensor([True, True, True, True, False, True])}
        for action, weight in (("heal", 3.), ("pearl", 5.)):
            mask = masks[action]
            expected = nn.functional.binary_cross_entropy_with_logits(
                pred[f"{action}_logits"][mask], targets[f"{action}_start"].ravel()[mask].float(),
                pos_weight=torch.tensor(weight),
            )
            torch.testing.assert_close(losses[action], expected)
            gradient, = torch.autograd.grad(losses[action], pred[f"{action}_logits"], retain_graph=True)
            self.assertEqual(gradient[~mask].abs().sum().item(), 0.)
            self.assertTrue((gradient[mask].abs() > 0).all())

        # Main state and opponent-pot losses must retain all original frames.
        multipliers = torch.tensor([1., 4., 1., 4., 1., 1.])
        ce = nn.functional.cross_entropy(pred["state_logits"], state, weight=CLASS_WEIGHTS, reduction="none")
        expected_state = (ce * multipliers).sum() / (CLASS_WEIGHTS[state] * multipliers).sum()
        torch.testing.assert_close(losses["state"], expected_state)
        torch.testing.assert_close(losses["pot_soon"], torch.tensor(2.).log())
        torch.testing.assert_close(losses["pot_urgency"], torch.tensor(.125))
        torch.testing.assert_close(total, sum(LOSS_WEIGHTS[name] * loss for name, loss in losses.items()))

    def test_all_continuation_batch_has_zero_loss_and_gradient_for_its_head(self):
        for state, action in ((1, "heal"), (2, "pearl")):
            pred, targets = self.inputs([state] * 3, [0] * 3, [0] * 3)
            _, losses, _ = self.losses(pred, targets)
            self.assertEqual(losses[action].item(), 0.)
            gradient, = torch.autograd.grad(losses[action], pred[f"{action}_logits"])
            torch.testing.assert_close(gradient, torch.zeros(3))

    def test_no_continuations_matches_original_weighted_bce(self):
        pred, targets = self.inputs([0, 1, 2], [0, 1, 0], [0, 0, 1])
        _, losses, _ = self.losses(pred, targets)
        for action, weight in (("heal", 3.), ("pearl", 5.)):
            expected = nn.functional.binary_cross_entropy_with_logits(
                pred[f"{action}_logits"], targets[f"{action}_start"].ravel().float(),
                pos_weight=torch.tensor(weight),
            )
            torch.testing.assert_close(losses[action], expected)


if __name__ == "__main__":
    unittest.main()
