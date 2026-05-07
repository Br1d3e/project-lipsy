from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd


PRE_TICKS = 20
POST_TICKS = 80
RETREAT_WINDOW_TICKS = 60
ANCHOR_COOLDOWN_TICKS = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate candidate labeling anchors from a Recorder v2 session folder."
    )
    parser.add_argument("session_dir", nargs="?", help="Session folder containing session_meta.json, events.jsonl, and frames.jsonl(.gz)")
    parser.add_argument("--output", "-o", help="Output anchors JSONL path. Defaults to <session_dir>/anchors.jsonl")
    return parser.parse_args()


def read_jsonl(path):
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".gz":
        return pd.read_json(path, lines=True, compression="gzip")
    return pd.read_json(path, lines=True)


def find_frames_path(session_dir):
    gz_path = session_dir / "frames.jsonl.gz"
    plain_path = session_dir / "frames.jsonl"
    if gz_path.exists():
        return gz_path
    if plain_path.exists():
        return plain_path
    raise FileNotFoundError("Missing frames.jsonl.gz or frames.jsonl")


def clean_value(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def lower_text(value):
    value = clean_value(value)
    if value is None:
        return ""
    return str(value).lower()


def tick_col(df):
    if "recordingTickIndex" in df.columns:
        return "recordingTickIndex"
    if "recording_tick_index" in df.columns:
        return "recording_tick_index"
    if "tick" in df.columns:
        return "tick"
    raise KeyError("No tick column found")


def event_tick(row):
    for key in ("recording_tick_index", "recordingTickIndex", "tick"):
        if key in row and pd.notna(row[key]):
            return int(row[key])
    return None


def event_detail(row, key, default=None):
    details = row.get("details", None)
    if isinstance(details, dict):
        return details.get(key, default)
    return default


def event_ticks(events, event_types):
    if events.empty or "type" not in events.columns:
        return []
    event_types = set(event_types)
    ticks = []
    for _, row in events.iterrows():
        if row.get("type") in event_types:
            tick = event_tick(row)
            if tick is not None:
                ticks.append(tick)
    return sorted(set(ticks))


def frame_at_or_before(frames, tick):
    if frames.empty:
        return None
    tick_name = tick_col(frames)
    subset = frames[frames[tick_name] <= tick]
    if subset.empty:
        return None
    return subset.iloc[-1]


def frame_at_or_after(frames, tick):
    if frames.empty:
        return None
    tick_name = tick_col(frames)
    subset = frames[frames[tick_name] >= tick]
    if subset.empty:
        return None
    return subset.iloc[0]


def series_bool(df, column):
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].fillna(False).astype(bool)


def series_numeric(df, column, default=np.nan):
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def item_contains(frames, needles):
    text = pd.Series("", index=frames.index, dtype="object")
    for column in ("myHeldItemType", "myHeldItem"):
        if column in frames.columns:
            text = text + " " + frames[column].fillna("").astype(str).str.lower()
    mask = pd.Series(False, index=frames.index)
    for needle in needles:
        mask = mask | text.str.contains(needle, regex=False)
    return mask


def item_text(frames):
    text = pd.Series("", index=frames.index, dtype="object")
    for column in ("myHeldItemType", "myHeldItem"):
        if column in frames.columns:
            text = text + " " + frames[column].fillna("").astype(str).str.lower()
    return text


def rising_edges(mask):
    previous = mask.shift(1, fill_value=False)
    return mask & ~previous


def item_use_segments(frames, item_mask, event_use_ticks=None):
    if frames.empty:
        return []

    event_use_ticks = set(event_use_ticks or [])
    tick_name = tick_col(frames)
    ticks = series_numeric(frames, tick_name).astype(int)
    use_mask = series_numeric(frames, "aUseItem", 0).fillna(0) > 0
    item_mask = item_mask.fillna(False).astype(bool)
    segments = []
    start_idx = None

    for idx in frames.index:
        if item_mask.loc[idx] and start_idx is None:
            start_idx = idx
        elif not item_mask.loc[idx] and start_idx is not None:
            segments.append(build_item_use_segment(frames, ticks, use_mask, event_use_ticks, start_idx, idx - 1))
            start_idx = None

    if start_idx is not None:
        segments.append(build_item_use_segment(frames, ticks, use_mask, event_use_ticks, start_idx, frames.index[-1]))

    return [segment for segment in segments if segment is not None]


def build_item_use_segment(frames, ticks, use_mask, event_use_ticks, start_idx, end_idx):
    segment_index = frames.loc[start_idx:end_idx].index
    start_tick = int(ticks.loc[start_idx])
    end_tick = int(ticks.loc[end_idx])
    frame_use_ticks = [int(ticks.loc[idx]) for idx in segment_index if bool(use_mask.loc[idx])]
    event_ticks_in_segment = sorted(tick for tick in event_use_ticks if start_tick <= tick <= end_tick)
    use_ticks = sorted(set(frame_use_ticks + event_ticks_in_segment))

    if not use_ticks:
        return None

    return {
        "start_tick": start_tick,
        "end_tick": end_tick,
        "tick": use_ticks[0],
        "use_ticks": use_ticks,
    }


def hp_value(row, prefix):
    if row is None:
        return None
    if prefix == "target":
        for column in ("targetEstimatedHp", "targetHp"):
            if column in row and pd.notna(row[column]):
                return float(row[column])
        return None
    column = "myHp"
    if column in row and pd.notna(row[column]):
        return float(row[column])
    return None


def any_tick_between(ticks, start_tick, end_tick):
    return any(start_tick <= tick <= end_tick for tick in ticks)


def first_tick_between(ticks, start_tick, end_tick):
    for tick in ticks:
        if start_tick <= tick <= end_tick:
            return tick
    return None


def target_using_item_between(frames, start_tick, end_tick):
    tick_name = tick_col(frames)
    window = frames[(frames[tick_name] >= start_tick) & (frames[tick_name] <= end_tick)]
    return bool(series_bool(window, "targetUsingItem").any()) if not window.empty else False


def anchor_dict(session_id, tick, kind, suggested_label, suggested_type=None,
                suggested_reward=None, start_offset=PRE_TICKS, end_offset=POST_TICKS,
                source=None, start_tick=None, end_tick=None):
    tick = int(tick)
    return {
        "session": session_id,
        "tick": tick,
        "start_tick": max(0, int(start_tick)) if start_tick is not None else max(0, tick - start_offset),
        "end_tick": int(end_tick) if end_tick is not None else tick + end_offset,
        "kind": kind,
        "suggested_label": suggested_label,
        "suggested_type": suggested_type,
        "suggested_reward": suggested_reward,
        "source": source,
    }


def add_anchor(anchors, session_id, tick, kind, suggested_label, suggested_type=None,
               suggested_reward=None, start_offset=PRE_TICKS, end_offset=POST_TICKS, source=None,
               cooldown=ANCHOR_COOLDOWN_TICKS, start_tick=None, end_tick=None):
    for anchor in anchors:
        if anchor["kind"] == kind and abs(anchor["tick"] - int(tick)) <= cooldown:
            return
    anchors.append(anchor_dict(
        session_id=session_id,
        tick=tick,
        kind=kind,
        suggested_label=suggested_label,
        suggested_type=suggested_type,
        suggested_reward=suggested_reward,
        start_offset=start_offset,
        end_offset=end_offset,
        source=source,
        start_tick=start_tick,
        end_tick=end_tick,
    ))


def generate_healing_anchors(frames, events, session_id, anchors):
    text = item_text(frames)
    health_pot_mask = text.str.contains("potion", regex=False) & (
        text.str.contains("splash", regex=False)
        | text.str.contains("instant health", regex=False)
        | text.str.contains("healing", regex=False)
    )
    for segment in item_use_segments(frames, health_pot_mask, event_ticks(events, ["POTION_USE"])):
        add_anchor(
            anchors, session_id, segment["tick"], "heal_candidate", "heal",
            source="held_potion_segment_use",
            start_tick=segment["start_tick"],
            end_tick=segment["end_tick"],
            cooldown=0,
        )


def generate_pearl_anchors(frames, events, session_id, anchors):
    pearl_segments = item_use_segments(
        frames,
        item_contains(frames, ["enderpearl", "ender_pearl", "pearl"]),
        event_ticks(events, ["PEARL_THROW"]),
    )
    attack_ticks = event_ticks(events, ["ATTACK_PACKET_SENT"])
    if "aAttack" in frames.columns:
        tick_name = tick_col(frames)
        attack_mask = series_numeric(frames, "aAttack", 0).fillna(0) > 0
        attack_ticks_from_frames = frames.loc[rising_edges(attack_mask), tick_name].tolist()
        attack_ticks = sorted(set(attack_ticks).union(int(t) for t in attack_ticks_from_frames))

    for segment in pearl_segments:
        tick = segment["tick"]
        start_tick = segment["start_tick"]
        end_tick = segment["end_tick"]
        attack_during_segment = any_tick_between(attack_ticks, start_tick, end_tick)
        row = frame_at_or_before(frames, tick)
        my_hp = hp_value(row, "my")
        target_hp = hp_value(row, "target")

        if attack_during_segment and my_hp is not None and target_hp is not None and my_hp > target_hp:
            suggested_type = "tryhard"
        elif attack_during_segment and target_using_item_between(frames, start_tick, end_tick):
            suggested_type = "agro"
        elif not attack_during_segment:
            suggested_type = "retreat"
        else:
            suggested_type = None

        add_anchor(
            anchors, session_id, tick, "pearl_candidate", "pearl",
            suggested_type=suggested_type,
            source="held_pearl_segment_use",
            start_tick=start_tick,
            end_tick=end_tick,
            cooldown=0,
        )


def generate_retreat_anchors(frames, session_id, anchors):
    if frames.empty:
        return
    tick_name = tick_col(frames)
    ticks = series_numeric(frames, tick_name)
    target_exists = series_bool(frames, "targetExists") if "targetExists" in frames.columns else pd.Series(True, index=frames.index)
    my_hp = series_numeric(frames, "myHp")
    target_distance = series_numeric(frames, "targetDistance")
    yaw_diff = series_numeric(frames, "yawDiffToTarget")
    hp_diff = series_numeric(frames, "hpDiff")

    low_hp = (my_hp <= 8.0) | (hp_diff <= -2.0)
    hard_yaw_away = yaw_diff.abs() >= 110.0
    distance_change = target_distance.diff(RETREAT_WINDOW_TICKS)
    distance_increasing = distance_change >= 2.0
    retreat_mask = target_exists & (low_hp | hard_yaw_away | distance_increasing)

    last_added = -10 ** 9
    for idx in frames.index[rising_edges(retreat_mask)]:
        tick = int(ticks.loc[idx])
        if tick - last_added < RETREAT_WINDOW_TICKS:
            continue
        add_anchor(
            anchors, session_id, tick, "retreat_candidate", "retreat",
            source="hp_yaw_or_distance_rule",
            start_offset=20,
            end_offset=RETREAT_WINDOW_TICKS,
            cooldown=RETREAT_WINDOW_TICKS,
        )
        last_added = tick


def generate_attack_anchors(frames, events, session_id, anchors):
    attack_ticks = set(event_ticks(events, ["ATTACK_PACKET_SENT"]))
    tick_name = tick_col(frames)

    if "aAttack" in frames.columns:
        attack_mask = series_numeric(frames, "aAttack", 0).fillna(0) > 0
        attack_ticks.update(int(t) for t in frames.loc[rising_edges(attack_mask), tick_name].tolist())

    for column in ("targetEstimatedLastDamage",):
        if column in frames.columns:
            damage_mask = series_numeric(frames, column, 0).fillna(0) > 0
            attack_ticks.update(int(t) for t in frames.loc[rising_edges(damage_mask), tick_name].tolist())

    if "mySwingProgress" in frames.columns:
        swing = series_numeric(frames, "mySwingProgress", 0).fillna(0)
        swing_start = (swing > 0) & (swing.shift(1, fill_value=0) <= 0)
        attack_ticks.update(int(t) for t in frames.loc[swing_start, tick_name].tolist())

    for tick in sorted(attack_ticks):
        add_anchor(
            anchors, session_id, tick, "attack_candidate", "attack",
            source="attack_packet_damage_or_swing",
            start_offset=10,
            end_offset=30,
            cooldown=10,
        )


def prepare_frames(frames):
    if frames.empty:
        return frames
    tick_name = tick_col(frames)
    frames = frames.sort_values(tick_name).reset_index(drop=True)
    if "targetExists" in frames.columns:
        target_exists = series_bool(frames, "targetExists")
        if "targetId" in frames.columns:
            target_id_exists = frames["targetId"].notna()
            frames = frames[target_exists & target_id_exists].reset_index(drop=True)
        else:
            frames = frames[target_exists].reset_index(drop=True)
    return frames


def write_anchors(path, anchors):
    with open(path, "w") as f:
        for i, anchor in enumerate(sorted(anchors, key=lambda a: (a["tick"], a["kind"]))):
            anchor = dict(anchor)
            anchor["id"] = "anc_%06d" % i
            cleaned = {key: clean_value(value) for key, value in anchor.items() if value is not None}
            f.write(json.dumps(cleaned, separators=(",", ":")) + "\n")


def main():
    args = parse_args()
    session_dir = Path(args.session_dir or input("Enter folder directory: ")).expanduser()
    meta_path = session_dir / "session_meta.json"
    events_path = session_dir / "events.jsonl"
    frames_path = find_frames_path(session_dir)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    events = read_jsonl(events_path)
    frames = prepare_frames(read_jsonl(frames_path))

    session_id = meta.get("sessionId") or meta.get("session_id") or session_dir.name
    print("Auto-generating label anchors for session %s" % session_id)

    anchors = []
    generate_healing_anchors(frames, events, session_id, anchors)
    generate_pearl_anchors(frames, events, session_id, anchors)
    generate_retreat_anchors(frames, session_id, anchors)
    generate_attack_anchors(frames, events, session_id, anchors)

    output_path = Path(args.output).expanduser() if args.output else session_dir / "anchors.jsonl"
    write_anchors(output_path, anchors)
    print("Wrote %d anchors to %s" % (len(anchors), output_path))


if __name__ == "__main__":
    main()
