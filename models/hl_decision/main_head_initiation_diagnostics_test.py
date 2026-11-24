import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.hl_decision.main_head_initiation_diagnostics import diagnose_main_head, predict_main_states
from models.torch_dataset import HLMultiTaskDataset


class MainHeadDiagnosticsTest(unittest.TestCase):
    def test_partition_and_annotation_precedence_for_both_actions(self):
        for action, action_id in [('heal', 1), ('pearl', 2)]:
            # Four separate pairs: captured, already active, wrong state,
            # and a new annotated episode within an unchanged true action.
            targets = {
                'state': [0, action_id, 0, action_id, 0, action_id, action_id, action_id],
                'round_id': [0, 0, 1, 1, 2, 2, 3, 3],
                'target_tick': [10, 11] * 4,
                'heal_start': [0] * 8, 'pearl_start': [0] * 8,
            }
            targets[f'{action}_start'] = [0, 1] * 4
            predicted = [0, action_id, action_id, action_id, 0, 0, 0, 0]
            result = diagnose_main_head(predicted, targets)['actions'][action]
            self.assertEqual(result['counts'], {
                'captured': 1, 'already_predicted_active': 1,
                'wrong_state_at_start': 1, 'annotation_boundary_mismatch': 1,
                'excluded_boundary_or_gap': 0,
            })
            self.assertEqual(result['missed_starts'], 3)
            self.assertEqual(result['recall'], .25)
            self.assertEqual(result['events'][-1]['round_id'], 3)
            self.assertFalse(result['events'][-1]['annotation_consistent'])

    def test_gaps_and_annotation_audit_do_not_change_exact_capture(self):
        targets = {'state': [1, 1, 0, 1, 1], 'round_id': [0] * 5,
                   'target_tick': [1, 3, 4, 5, 6],
                   'heal_start': [1, 1, 0, 0, 1], 'pearl_start': [0] * 5}
        report = diagnose_main_head([0, 0, 0, 0, 1], targets)['actions']['heal']
        self.assertEqual(report['counts']['excluded_boundary_or_gap'], 2)
        self.assertEqual(report['counts']['captured'], 1)
        self.assertEqual(report['starts_without_true_state_edge'], 1)
        self.assertEqual(report['true_state_edges_without_start'], [{'round_id': 0, 'target_tick': 5}])
        self.assertIsNone(report['events'][1]['previous_predicted_state'])

    def test_empty_and_length_mismatch(self):
        targets = {name: [] for name in ('state', 'round_id', 'target_tick', 'heal_start', 'pearl_start')}
        self.assertIsNone(diagnose_main_head([], targets)['actions']['heal']['recall'])
        with self.assertRaises(ValueError):
            diagnose_main_head([0], targets)

    def test_inference_retains_order_across_batches_and_supports_both_outputs(self):
        class Model(nn.Module):
            def __init__(self, as_dict):
                super().__init__()
                self.as_dict = as_dict

            def forward(self, x):
                logits = torch.nn.functional.one_hot(x[:, 0, 0].long(), 6).float()
                return {'state_logits': logits} if self.as_dict else logits

        data = HLMultiTaskDataset(np.array([0, 1, 2, 1, 0]).reshape(5, 1, 1), {'state': [0] * 5})
        for as_dict in (False, True):
            predicted = predict_main_states(Model(as_dict), DataLoader(data, batch_size=2), 'cpu')
            np.testing.assert_array_equal(predicted, [0, 1, 2, 1, 0])


if __name__ == '__main__':
    unittest.main()
