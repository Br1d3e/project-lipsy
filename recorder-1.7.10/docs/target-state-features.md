# Target-state collection, recorder v3

Recording metadata uses `recorderVersion: v3` to distinguish the changed state
semantics from earlier recordings. Training and the 1.8.9 inference protocol
have not yet migrated. `targetEating` and `targetSpeeding` remain as holding aliases; `targetPotting`
now means detected splash healing (see below). Old
recordings retain their original active-use semantics: renaming their columns
alone does not make the datasets equivalent.

| Canonical field | Compatibility aliases | True when |
| --- | --- | --- |
| `targetHoldingFood` | `targetEating` | Target holds an item recognized as food |
| `targetHoldingSpeedPotion` | `targetSpeeding` | Target holds an item recognized as a speed potion |
| `targetHoldingHealPotion` | `targetHealing` when provided by the recording version | Target holds a splash healing potion |

All these fields are null without a target. None requires `isUsingItem()`.
Holding a potion does not reset the hit proxy. `targetPearling` now means a newly detected enemy pearl throw (see v5 below).

## `hitsBeforeTargetHeals`

A nonnegative count of inferred melee hits by us since the last detected target
healing-potion throw or healing splash affecting the target. This is a running
count, not a prediction of how many additional hits the target can survive.

- Starts at zero each manual round; exported as null while the target is absent.
  The stored count survives visibility gaps and entity-ID changes for the same round opponent.
- An increase in target hurtTime counts only if an outgoing attack packet to
  that same target was seen within the preceding five recording ticks, inclusive.
  Each pending attack can confirm at most one hurt-time increase. Later attacks
  replace the pending attack timestamp.
- Damage to ourselves does not reset the counter; this differs from myCombo.
- Resets when the inventory tracker detects a new target-owned healing-potion
  projectile, even when estimated potion inventory is already zero. It reuses
  the existing thrower/proximity attribution and projectile-ID deduplication.
- Also resets when the existing HP tracker matches a healing splash and its
  existing geometry calculation gives positive healing, including our splashes.
  This can reset again at impact after a reset at throw time.
- Estimated natural regeneration does not reset it. That is simulated rather
  than an observed heal, and using it would couple the proxy back to HP/food drift.
- Entity-ID changes, target loss, and round/recording restart clear pending attacks.
  Only round/recording restart clears the accumulated count.

The HP estimator's damage, healing, and regeneration calculations are unchanged.
No damage amount or healing amount is accumulated into this counter. However,
positive splash detection still depends on the existing heal geometry. The
counter does not remove event-detection uncertainty.

Client damage notifications lack the attacker identity. The five-tick correlation
is a heuristic: pearl/fall/third-party damage coincident with our attack may count,
and delayed confirmations can be missed. Target potion ownership is also inferred
when the projectile's thrower is unavailable. This is not authoritative HP data.

## Assessment

The proxy is useful as an additional measure of recent combat pressure, but is
not a calibrated HP estimate and does not currently modify targetEstimatedHp.
Its value can increase even if we take return hits, unlike a combo counter.

| Observations | Interpretation supported by the proxy |
| --- | --- |
| High count, low estimated HP, target holding healing potion | Multiple signals support a possible heal/retreat decision; no guaranteed threshold |
| High count, estimated HP still high | Estimator disagreement worth investigating; could be missed healing, wrong damage attribution, or HP drift |
| Low count after detected healing | Recent reset; recovery is possible, but remaining HP is unknown |
| Low count after a thrown potion that misses | False reassurance: use does not guarantee recovery |
| Zero at round start | No observed history; does not establish full HP |

A target resuming pressure also depends on distance, movement, resources, and our
HP. A low count alone cannot distinguish recovery from a missed potion or a newly
selected target. Do not force an attack/retreat rule from this counter.

No new recording dataset was supplied for empirical validation. To measure its
incremental value, record controlled duels with independently logged target HP,
potion throws/impacts, and target actions. Compare otherwise identical models
with and without the proxy, keeping entire matches separated between train and
test. Evaluate low-HP detection and subsequent target heal/retreat versus attack
behavior, including missed potions, trade hits, pearl landings, target switches,
and delayed packets. Report event attribution precision/recall as well as model
performance. Server or paired-client truth is for evaluation, not an input that
would be unavailable during deployment.

## Standalone counter regression checks

From the recorder directory, compile `TargetHitTracker.java` and
`src/test/java/com/bot/recorder/targetTracker/TargetHitTrackerTest.java` into a
temporary directory with Java 8, then run
`com.bot.recorder.targetTracker.TargetHitTrackerTest`. These checks cover counter
transitions; they do not simulate Minecraft networking or validate attribution.

## Manual round boundaries

Recording starts with `recorderRoundId = 1`. Press **Backspace** (rebindable as
**Next recorder round** under Project Lipsy controls) to increment it and clear
round tracking. Frames include `recorderRoundId`; events include it in `details`.
A `ROUND_START` event identifies recording start or a manual hotkey boundary.
The recording tick index remains monotonic within the session.

Press the key at the beginning of each match, when the intended opponent is the
nearest visible player. The recorder locks that player's UUID until the next
manual round. Another player cannot silently replace an out-of-range opponent.
The same UUID can be used again in the next round; the round ID distinguishes it.
If the key is pressed in a lobby with bystanders, the nearest bystander can become
the locked target; press it again at the actual match start.

A visibility gap preserves HP, hunger/saturation/exhaustion, potion/pearl resource
estimates, combos, hit count, and enemy throw timer/cooldown. Estimators do not
invent unseen hits, throws, or healing. Observation histories and pending hits
are cleared; reacquisition establishes hurt-time and movement baselines rather
than treating a visibility jump as a hit or movement exhaustion. Rotation deltas
restart after the gap. A new entity ID for the locked UUID rebinds the trackers
without resetting round estimates.

The hotkey resets both players' recorder timers/cooldowns, all target estimators,
combos, seen/active projectile IDs, pending uses/attacks, and angle/cache history.
Inventory and spatial caches refresh on the next recorded frame. It does not
alter actual Minecraft health, inventory, or server cooldowns. `lastBlockTick`
is reset here and at recording start, preventing negative elapsed block times.
Self-damage tracking remains conditional on a visible target.

Self pearl cooldown uses a per-tick inventory sample across all 36 inventory
slots. A use attempt starts cooldown only if it was zero. A decrease of exactly
one pearl also starts/restarts it, including during cooldown; repeated attempts
without consumption only count it down. The consumption condition is not proof
of a successful server teleport: dropping one pearl or client prediction can
also decrease inventory. Multiple uses between samples cannot be reconstructed
from a net decrease larger than one.

Training and inference migration remains deferred. Consumers should group by
session plus `recorderRoundId` before constructing sequence windows, previous
state labels, or forward-filled values. A visibility gap is not a round boundary.

## Auxiliary combat and movement features (v3)

`features/CombatHistoryTracker` owns pressure and hit history;
`features/TargetMovementTracker` owns position history and geometry.
The existing target HP damage/heal calculations are not changed by these features.

| Recorded field | Meaning |
| --- | --- |
| `myHitPressure` | Decaying count of inferred hits we land; target splash healing resets it |
| `targetHitPressure` | Decaying count of incoming hurt events; an observed increase in our HP resets it |
| `hitsLandedLastK` | Inferred landed hits in ticks `(t-K, t]` |
| `hitsTakenLastK` | Incoming hurt events in the same window |
| `netHitsLastK` | Landed minus taken; zero can mean equal trades or no combat |
| `ticksSinceCombat` | Ticks since the latest event from either hit stream; null before any event |
| `targetRadialVelocity` | Horizontal relative velocity projected along self-to-target direction; blocks/tick |
| `targetOutwardVelocity` | Target's horizontal velocity along that direction, excluding self movement; blocks/tick |
| `targetFacingMeCos` | +1 facing us, 0 sideways, -1 away; null at zero horizontal separation |
| `distanceDelta5`, `distanceDelta20` | Current horizontal separation minus separation 5/20 ticks ago; blocks |
| `targetPotting` | One-frame pulse after a matched splash estimated to heal the target; null without a visible target |

`targetPotting` is no longer an alias for holding a potion. The pulse uses the
existing TargetHpTracker impact matching and positive-healing geometry test. It
can include our own splash. It is not an authoritative server heal confirmation,
or proof that the target threw a potion. An unmatched/missed impact does not
produce a pulse. Several impacts between recorded frames collapse into one true
frame. Holding, throwing without a matched heal, and estimated natural regeneration
do not reset outgoing hit pressure. The older hitsBeforeTargetHeals proxy still
also resets on an inferred enemy potion throw; those semantics are unchanged.

Landed-hit events use the existing five-tick attack-packet/hurt-time correlation,
shared with hitsBeforeTargetHeals. Incoming events retain the existing assumption
that self hurt-time increases while the target is visible come from that opponent.
Old combo/HP-estimator hit attribution is unchanged. Heal resets leave rolling
hit counts and last-combat time intact. Self healing uses a positive net HP
change between samples, so simultaneous healing and damage can conceal recovery;
other HP increases can also trigger a reset.

Pressure decays before adding a tick's hit, using `decay ** elapsedTicks`.
`config/projectlipsy.cfg` exposes `features.hitPressureDecay` (default 0.95,
range 0..1) and `features.combatHistoryTicks` (default 60, range 1..1200).
Set values before launching Minecraft; live configuration reload is not provided.
The RECORD_START event records both settings and the five-tick velocity window.
Different K/decay settings should not silently be mixed in training.

Velocity uses position differences over five continuous recording ticks, projected
onto the current self-to-target unit vector. Distance deltas also use horizontal
separation, whereas the existing targetDistance field is three-dimensional.
Movement history resets on visibility gaps, entity-ID changes, position-correction/
teleport packets, and a horizontal displacement over four blocks in one tick by
either player. A reset produces null derivatives until enough new history exists.
Small unannounced position jumps remain a limitation. Facing needs only the
current position/yaw observation and is available before velocity warms up.

Round reset clears all auxiliary history and pressures. Visibility loss only
clears movement history: combat pressure continues to decay and observed hits
age out naturally. Zero counts during incomplete observation do not prove that
no unseen combat occurred. Training/inference migration remains deferred.

## Direct fields and playstyle samples (v5)

- `refillableHealthPotions`: total inventory healing potions minus hotbar healing
  potions, clamped at zero. Uses the same five-tick inventory snapshot as those
  existing counts and the same item classifier.
- `myItemUseTicks`: the local client's elapsed continuous item-use duration;
  zero while idle. This includes sustained sword blocking and drinking/eating,
  not just healing. An instantaneous splash or pearl throw has no sustained use.
- `targetHoldingSword`: holding-only boolean, null while the target is absent.
- `targetPearling`: true only on a frame with a newly detected enemy pearl entity;
  false on later frames with that same pearl, null without a visible target.
  It shares the existing thrower/proximity attribution and projectile-ID
  deduplication used for enemy cooldown and ENEMY_PEARL_THROW events. It does not
  confirm landing or teleport success. Multiple throws in a frame produce one
  true frame; each detected entity still produces its own throw event.

`TargetPlaystyleTracker` owns round-local samples and running aggregates:

| Frame field | Meaning |
| --- | --- |
| `meanHitsBeforeTargetHeals` | Mean accepted hit-cycle sample |
| `medianHitsBeforeTargetHeals` | Median accepted hit-cycle sample |
| `targetHealEpisodeSampleCount` | Number of accepted hit-cycle samples |
| `meanTargetPearlGapTicks` | Mean completed observed pearl interval, including cooldown |
| `medianTargetPearlGapTicks` | Median completed observed pearl interval |
| `targetPearlIntervalSampleCount` | Number of accepted pearl intervals |

Means/medians are null until a valid sample exists; counts start at zero. They
remain available during visibility loss but reset on the manual round key.

Healing episodes start at the first newly attributed target healing-potion throw.
Throws separated by at most 20 ticks belong to the same episode (measured from
its most recent throw). At the first throw, TARGET_HEAL_EPISODE_SAMPLE records
`hitsBeforeHealing`, `completeObservation`, target entity ID, and tick. This
closes the preceding hit cycle immediately, without future data. Additional
throws in that episode do not produce extra samples; each throw restarts the
hit-cycle counter, so the next episode measures hits since the preceding
episode's last throw. The 20-tick threshold is an explicit grouping heuristic,
recorded as `healEpisodeGapTicks` in RECORD_START.

The episode counter uses the same attack-correlated landed-hit events as combat
history, independently of the older frame-level hitsBeforeTargetHeals counter.
That older counter also resets on splash impacts, including our splashes; the
playstyle counter intentionally does not. Neither a splash impact nor merely
holding a potion creates a playstyle sample. This prevents throw/impact double
counting and does not mistake a self-thrown heal for an opponent decision.
A target throw that misses still counts as a healing decision, not successful
recovery. Ownership inference can still misattribute close-range potions.

Visibility loss invalidates the unfinished hit cycle and pearl interval. The
first healing episode after a gap is logged with null hits and
`completeObservation=false`, excluded from aggregates; its throw establishes a
new baseline. The first pearl after a gap establishes a new interval baseline.
Previously accepted samples survive. Positive pearl gaps produce
TARGET_PEARL_INTERVAL_SAMPLE events with `gapTicks`, target/pearl entity IDs,
and tick. First throws and same-tick pairs do not create interval samples.
Short positive gaps are retained as observed rather than clamped to 320;
missed throws can still inflate intervals. Round-end unfinished intervals never
become samples. These features describe observed behavior, not calibrated HP or
an intrinsic opponent personality. Training/inference migration remains deferred.
