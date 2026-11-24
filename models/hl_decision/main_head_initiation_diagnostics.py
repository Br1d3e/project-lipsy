"""Audit exact-tick main-state initiation misses against canonical label_start targets.

Run from the repository root:
    python -m models.hl_decision.main_head_initiation_diagnostics

Edit MODEL_FILENAME, SPLIT, and OUTPUT_PATH below before running.

The three miss categories partition missed starts. Annotation inconsistencies take
precedence over prediction errors within that partition. First endpoints and gaps
are excluded, matching TrainTorchModel.score_initiation; no labels are shifted.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.hl_decision.hl_decision_datasets import HLDecisionDataset, TARGETS
from models.hl_decision.train_hl_decision_model import TrainTorchModel, SEQ_LEN, BATCH_SIZE
from models.torch_dataset import HLMultiTaskDataset


MODEL_FILENAME = "hl_multihead_20_test_heal.pt"
SPLIT = "val"  # train, val, or test
OUTPUT_PATH = None  # Optional Path for the per-start JSON report.


def diagnose_main_head(predicted_states, targets):
    """Return counts and identifiable start records; all inputs share endpoint order."""
    predicted = np.asarray(predicted_states).reshape(-1)
    state = np.asarray(targets['state']).reshape(-1)
    rounds = np.asarray(targets['round_id']).reshape(-1)
    ticks = np.asarray(targets['target_tick']).reshape(-1)
    starts = {action: np.asarray(targets[f'{action}_start']).reshape(-1).astype(bool)
              for action in ('heal', 'pearl')}
    if any(values.shape != predicted.shape for values in (state, rounds, ticks, *starts.values())):
        raise ValueError('Predictions, labels, and endpoint metadata must have matching lengths')

    contiguous = np.zeros(len(state), dtype=bool)
    contiguous[1:] = (rounds[1:] == rounds[:-1]) & (ticks[1:] == ticks[:-1] + 1)
    result = {'sample_count': len(state), 'contiguous_count': int(contiguous.sum()), 'actions': {}}
    for action, truth in starts.items():
        action_id = TARGETS[action]
        active = predicted == action_id
        predicted_edge = np.zeros(len(state), dtype=bool)
        predicted_edge[1:] = active[1:] & ~active[:-1]
        true_edge = np.zeros(len(state), dtype=bool)
        true_edge[1:] = (state[1:] == action_id) & (state[:-1] != action_id)

        captured = truth & contiguous & predicted_edge
        missed = truth & contiguous & ~predicted_edge
        inconsistent = truth & contiguous & ~true_edge
        boundary_miss = missed & inconsistent
        wrong_state = missed & ~inconsistent & ~active
        already_active = missed & ~inconsistent & active
        categories = {
            'captured': captured,
            'annotation_boundary_mismatch': boundary_miss,
            'wrong_state_at_start': wrong_state,
            'already_predicted_active': already_active,
            'excluded_boundary_or_gap': truth & ~contiguous,
        }
        events = []
        for index in np.flatnonzero(truth):
            events.append({
                'round_id': int(rounds[index]), 'target_tick': int(ticks[index]),
                'category': next(name for name, mask in categories.items() if mask[index]),
                'true_state': int(state[index]), 'predicted_state': int(predicted[index]),
                'previous_true_state': int(state[index - 1]) if contiguous[index] else None,
                'previous_predicted_state': int(predicted[index - 1]) if contiguous[index] else None,
                'annotation_consistent': bool(true_edge[index]) if contiguous[index] else None,
            })
        count = int((truth & contiguous).sum())
        extra_true_edges = contiguous & true_edge & ~truth
        result['actions'][action] = {
            'total_starts': int(truth.sum()), 'evaluated_starts': count,
            'counts': {name: int(mask.sum()) for name, mask in categories.items()},
            'missed_starts': int(missed.sum()),
            'recall': int(captured.sum()) / count if count else None,
            # Audit even inconsistencies that happened to be captured by the model.
            'starts_without_true_state_edge': int(inconsistent.sum()),
            'true_state_edges_without_start': [
                {'round_id': int(rounds[i]), 'target_tick': int(ticks[i])}
                for i in np.flatnonzero(extra_true_edges)
            ],
            'events': events,
        }
    return result


def predict_main_states(model, loader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in loader:
            output = model(x.to(device))
            logits = output['state_logits'] if isinstance(output, dict) else output
            if logits.ndim != 2 or logits.shape[1] != len(TARGETS):
                raise ValueError('Diagnostics require a six-state model output')
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.int64)


def evaluate_checkpoint(checkpoint, split='val'):
    model = TrainTorchModel.load_model(checkpoint)
    metadata = getattr(model, 'checkpoint_meta', {})
    dataset = HLDecisionDataset(use_cache=True, build_cache=False, features=metadata.get('features'))
    x, targets = getattr(dataset, f'get_{split}_X_Y')(
        seq_len=metadata.get('seq_len', SEQ_LEN), return_auxiliary_targets=True, msg=False,
    )
    loader = DataLoader(HLMultiTaskDataset(x, targets), batch_size=BATCH_SIZE, shuffle=False)
    predictions = predict_main_states(model, loader, TrainTorchModel.get_device())
    report = diagnose_main_head(predictions, targets)
    report.update(checkpoint=str(Path(checkpoint).resolve()), split=split)
    return report


def main():
    checkpoint = Path(__file__).resolve().parent / 'models' / MODEL_FILENAME
    report = evaluate_checkpoint(checkpoint, SPLIT)
    print(f"\n{report['split']}: {report['checkpoint']}")
    print(f"Endpoints: {report['sample_count']}; contiguous: {report['contiguous_count']}")
    for action, data in report['actions'].items():
        print(f"\n{action}: {data['evaluated_starts']} evaluated starts; recall={data['recall']}")
        for name, count in data['counts'].items():
            print(f"  {name}: {count}")
        print(f"  starts_without_true_state_edge: {data['starts_without_true_state_edge']}")
        print(f"  true_state_edges_without_start: {len(data['true_state_edges_without_start'])}")
    if OUTPUT_PATH is not None:
        Path(OUTPUT_PATH).write_text(json.dumps(report, indent=2) + '\n')
        print(f"Saved per-start diagnostics: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
