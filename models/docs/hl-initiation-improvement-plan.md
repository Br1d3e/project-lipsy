# HL action initiation improvement plan

Updated: 2026-10-09
Status: step 1 implemented and regression-tested; steps 2–6 remain pending.

## Objective and fixed decisions

Improve heal/pearl initiation while preserving imitation of actual human gameplay.
High continuation accuracy is insufficient evidence of good initiation. State-free
experiments have not yet established whether the limitation is supervision,
optimization, missing observations, or generalization.

- Preserve exact next-tick semantics: observations through tick `t`, state and
  action-start targets at `t+1`. Do not add another shift or widen positive labels
  into a future horizon.
- Existing `heal_start` and `pearl_start` auxiliary targets derived directly from
  `label_start` are authoritative for initiation precision/recall, including main
  head evaluation. Do not reconstruct ground truth from state-class rising edges.
- Preserve recorded actions and their resulting trajectories. Do not insert
  retrospective alternative actions or synthesize their future features.
- Keep the main state loss on the original frames. Initiation masks and weights
  apply locally to initiation losses, not to the combined multitask loss or global
  sampling distribution.
- Distinguish offline human-event metrics from runtime behavior. Runtime traces
  alone do not supply expert ground truth for missed starts.

## Ordered work

### 1. Repair evaluation and checkpoint selection together

Evaluation defines which checkpoint is best, so finish these repairs before
comparing new training runs.

- Align both heads' initiation evaluation with the existing start-target arrays
  at the same sequence endpoints. Preserve session, round, and tick identity.
- Define prediction rules explicitly: main-head action selection and auxiliary
  threshold decisions are different outputs. Compare them on the same eligible
  frames against the same start labels. If predicted rising-edge/event metrics
  are also retained, report them separately and never cross rounds or tick gaps.
- Reconcile differing positive counts before interpreting precision/recall.
- Independently track minimum validation loss, main-head heal-start F1 improvement,
  and main-head pearl-start F1 improvement. Replace mutually exclusive best-metric
  updates and shared-file overwriting with independent checkpoint files.
- Save the epoch and complete metrics for each checkpoint, plus architecture,
  feature order, preprocessing, loss settings, sequence length, split identity,
  and seed. Report restored-checkpoint metrics separately from historical bests.
- Select initiation checkpoints by F1 (the harmonic mean of precision and recall),
  and retain precision, recall, and false-positive counts in their reports.
- Verify validation/test loader routing, including the current `data_loader`
  versus `dataloader` keyword discrepancy. Tune on validation, not test.

Verification: a known short sequence gives identical ground-truth start counts
for both heads; round boundaries do not create events; independently improving
metrics save distinct recoverable checkpoints whose reported metrics reproduce.

Implemented 2026-10-09:

- `<name>_loss.pt`, `<name>_heal.pt`, and `<name>_pearl.pt` update independently,
  selecting heal/pearl winners by main-head initiation F1, including an initial
  checkpoint when F1 is zero. Patience counts epochs
  without any improvement. The in-memory model restores the minimum-loss winner;
  historical best epochs/metrics are reported separately for each criterion.
- Checkpoints store epoch, validation metrics (counts, precision/recall, losses,
  AP, state confusion matrix, regression metrics), training loss summaries,
  architecture configuration, preprocessing, feature order, loss settings, seed,
  and SHA-256 identities of the exact training/validation windows and targets.
  Reloading supports the existing CNN, linear, and LSTM checkpoint layouts.
- Dataset outputs carry integer `round_id` and `target_tick` metadata; round IDs
  identify distinct source rounds across sessions within each split. Metadata is
  not a model feature. Row pairs separated by a recording gap cannot form a
  next-tick target.
- The matched initiation comparison uses `label_start`-derived truth at `t+1`.
  Main predictions are rising edges of argmax action selections; auxiliary
  predictions are scores >= 0.5, without differencing the start labels again.
  Both are scored on the same consecutive-tick, same-round endpoints. The first
  prediction in each segment has no known preceding prediction and is excluded
  from this comparison, with excluded samples/positives reported. Full-endpoint
  auxiliary metrics remain available separately. This continuity filter is not
  the action-possibility loss mask planned in step 3.
- `initiation_baseline.py` now delegates to the training evaluator. The old
  tolerance-based episode report was removed rather than mixing another event
  definition into checkpoint selection. Run it as
  `python3 -m models.hl_decision.initiation_baseline` from the repository root.
- Test-loader routing is corrected. Test evaluation uses checkpoint feature order
  and sequence length; post-training evaluation is explicitly labelled validation.

Regression command (from repository root):

```sh
MPLCONFIGDIR=/tmp/lipsy-mpl python3 -m unittest models.hl_decision.evaluation_checkpoints_test models.hl_decision.test_shap_adapter models.hl_decision.test_feature_transforms models.hl_decision.test_socket_preprocessing -v
```

Checks cover canonical starts, consecutive start labels, gaps/round boundaries,
integer endpoint precision, independent checkpoint winners, patience, architecture
reloads, saved-metric reproduction, and test-loader routing. These are synthetic
regression checks, not a full training run or evidence of improved model quality.

### 2. Establish a reproducible baseline

Use one fixed LSTM configuration and unchanged preprocessing, loss settings, and
data split initially. Record training and validation initiation metrics, overall
state metrics, AP, and positive/negative counts. Reuse sequence endpoints across
ablations. Repeat promising comparisons with multiple seeds.

Optional small-set fitting diagnostic: select a small fixed set of distinct human
heal/pearl starts and matched non-start windows; fit each binary task independently
and evaluate on those same windows. Keep `SEQ_LEN` fixed. `transition_context`
may select examples but must not broaden the positive target. This deliberately
tests memorization, not generalization. Start with a roughly balanced subset;
retain natural distributions for full validation/testing.

### 3. Add and audit initiation possibility masks

Define a separate eligibility mask `m_a(t)` for each action using only information
available through `t`. Start with whether another instance of the action can be
initiated; audit action/episode semantics before excluding active-action frames.
Add inventory/cooldown constraints only after verifying recorder/runtime parity.
Possibility means executable, not a hand-written judgment that an action is wise.

For each initiation head use unreduced weighted BCE and reduce over eligible
examples: `sum(m * BCE) / sum(m)`. A batch with no eligible examples contributes
zero for that initiation task. Leave the main state and other auxiliary losses
unchanged. Compare masked and unmasked runs while holding other factors fixed.

Audit masked-out positive starts explicitly; investigate timing, consecutive
same-action episodes, or availability mismatches rather than silently deleting
them. Eligibility must not use future target values. Report both coverage and
eligible-set metrics so excluding examples cannot masquerade as an improvement.
Use the same eligibility semantics in deployment before enabling these heads.

### 4. Isolate task and state-context effects

Run controlled comparisons before a broader architecture search:

1. Independent heal-start and pearl-start models on the 65 state-free features,
   without shared multitask losses. Compare against the multitask baseline using
   the same targets, eligible endpoints, and binary weighting where applicable.
2. Six-state model without auxiliary losses. Explicitly document whether auxiliary
   predictions remain connected to the state head; removing losses and removing
   those connections are different ablations and must not be conflated.
3. State-context factorial: 65 features; add only six `s_*` channels; add only
   `ticksInCurrentState`; add both. Compare starts separately from continuation.
   Verify that any retained state context is causal and reproducible at runtime.

Interpretation: successful tiny-set fitting but poor held-out results points
toward generalization/coverage; better independent triggers implicate task
competition; state-context improvements require checking whether they help starts
or merely continuation. None alone proves leakage or insufficient features.

### 5. Improve architecture using the preceding evidence

- Retain the LSTM as a reference rather than assuming a larger model is needed.
- A `Linear` encoder applied per tick followed by selecting the last tick is not
  a temporal MLP. Use an explicitly defined nonlinear baseline if testing MLPs.
- The tested three-tick convolution plus whole-window average pooling weakens
  recency information. A temporal alternative should preserve the final-tick
  representation and a defined causal history, rather than averaging away when
  an observation occurred. A direct last-frame feature path is another controlled
  experiment, not a required simultaneous change.
- If task separation helps, consider distinct initiation and execution-context
  paths. Do not introduce more auxiliary tasks or larger positive weights without
  evidence that they address the measured failure.

Compare architectures under the same corrected metrics, masks, data, and selection
rules. Save architecture identity so checkpoints do not depend on whichever
encoder happens to be selected in the source file.

### 6. Later: deployment corrections / DAgger-style collection

Deferred until offline evaluation and training comparisons are reliable. Verify
runtime feature parity and log observations, predictions, executed states, and
overrides. Collect human corrections on states actually visited by the bot,
preferably through real-time takeover with the actual resulting trajectory.
Mark takeover boundaries and retain the original observations/history. Aggregate
these examples for later training; do not rewrite earlier trajectories or invent
counterfactual futures. This is a later collection experiment, not implemented
DAgger or an immediate deployment commitment.

## Deferred outside this plan

Retrospective acceptable-action relabeling, stochastic action selection, altered
prediction horizons, simulated future trees, and additional auxiliary objectives.
Human variability remains a possible limitation, not an established explanation
for the current initiation failures.

## Relevant implementation files

- `models/hl_decision/hl_decision_datasets.py`: `label_start`, start targets,
  sequence alignment, feature subsets, future eligibility metadata.
- `models/hl_decision/train_hl_decision_model.py`: checkpoint selection, losses,
  evaluation, architecture, experiment settings.
- `models/hl_decision/initiation_baseline.py`: align diagnostic metrics with the
  same canonical start targets and continuity rules.
- `models/socket_server.py` and `bot-1.8.9/`: later runtime parity and corrections.
