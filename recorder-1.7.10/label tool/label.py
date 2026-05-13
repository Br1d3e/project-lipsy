import json
import math
import shlex
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_FIELDS = [
    "id",
    "session",
    "start_tick",
    "end_tick",
    "label",
    "type",
    "reward",
    "source",
    "reason",
    "failure_reason",
]

EDITABLE_FIELDS = [
    "start_tick",
    "end_tick",
    "label",
    "type",
    "reward",
    "source",
    "reason",
    "failure_reason",
]

FIELD_ALIASES = {
    "l": "label",
    "label": "label",
    "t": "type",
    "type": "type",
    "r": "reward",
    "reward": "reward",
    "rs": "reason",
    "reason": "reason",
    "fr": "failure_reason",
    "failure_reason": "failure_reason",
    "st": "start_tick",
    "start_tick": "start_tick",
    "et": "end_tick",
    "end_tick": "end_tick",
    "src": "source",
    "source": "source",
}

cursor_id = 0

REVIEW_PRESETS = {
    "pearl": {
        "1": ("tryhard", 1),
        "2": ("tryhard", 0),
        "3": ("tryhard", -1),
        "4": ("agro", 1),
        "5": ("retreat", 1),
        "6": ("side", 1),
    },
    "heal": {
        "1": ("fast", 1),
        "2": ("turn", 1),
        "3": ("front", 1),
        "4": ("miss", -1),
    },
    "retreat": {
        "1": ("turn", 1),
        "2": ("jump", 1),
        "3": ("s", 1),
        "4": ("failed_retreat", -1),
    },
}


def read_jsonl(path):
    if not path.exists():
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        if path.suffix == ".gz":
            return pd.read_json(path, lines=True, compression="gzip")
        return pd.read_json(f, lines=True)



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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def reward_is_failure(value):
    value = clean_value(value)
    if value is None:
        return False
    try:
        return float(value) == -1.0
    except (TypeError, ValueError):
        return str(value).strip() == "-1"


def resolve_field_name(field):
    return FIELD_ALIASES.get(str(field).strip().lower())


def normalize_label(label, session_id):
    normalized = {field: clean_value(label.get(field)) for field in LABEL_FIELDS}
    old_notes = clean_value(label.get("notes"))
    if old_notes is not None:
        if reward_is_failure(normalized.get("reward")) and normalized.get("failure_reason") is None:
            normalized["failure_reason"] = old_notes
        elif normalized.get("reason") is None:
            normalized["reason"] = old_notes
    normalized["session"] = normalized["session"] or session_id
    normalized["source"] = normalized["source"] or "manual"
    return normalized


def labels_to_dataframe(labels, drop_session=True):
    rows = [labels[label_id] for label_id in sorted(labels.keys(), key=label_sort_key)]
    df = pd.DataFrame(rows, columns=LABEL_FIELDS)
    if drop_session and "session" in df.columns:
        df = df.drop(columns=["session"])
    return df


def label_sort_key(label_id):
    text = str(label_id)
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else text


def resolve_label_id(labels, value):
    text = str(value).strip()
    if text in labels:
        return text
    if text.isdigit():
        numeric = int(text)
        for label_id in labels.keys():
            if label_sort_key(label_id) == numeric:
                return label_id
    return None


def flatten_label_ids(label_ids):
    flattened = []
    for item in label_ids:
        if isinstance(item, (list, tuple, set)):
            flattened.extend(flatten_label_ids(item))
        elif item is not None:
            flattened.append(item)
    return flattened


def unique_label_ids(label_ids):
    unique = []
    for label_id in flatten_label_ids(label_ids):
        if label_id not in unique:
            unique.append(label_id)
    return unique


def next_label_id(labels):
    max_id = 0
    for label_id in labels.keys():
        text = str(label_id)
        if text.startswith("auto_"):
            continue
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            max_id = max(max_id, int(digits))
    return "lb_%05d" % (max_id + 1)


def next_auto_label_id(labels):
    max_id = 0
    for label_id in labels.keys():
        text = str(label_id)
        if not text.startswith("auto_"):
            continue
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            max_id = max(max_id, int(digits))
    return "auto_%05d" % (max_id + 1)


def anchor_source(anchor):
    anchor_id = clean_value(anchor.get("id"))
    source = clean_value(anchor.get("source"))
    if anchor_id is None:
        return "anchor" if source is None else "anchor:%s" % source
    return "anchor:%s" % anchor_id if source is None else "anchor:%s:%s" % (anchor_id, source)


def label_from_anchor(anchor, session_id, label_id):
    return normalize_label({
        "id": label_id,
        "session": anchor.get("session") or session_id,
        "start_tick": anchor.get("start_tick"),
        "end_tick": anchor.get("end_tick"),
        "label": anchor.get("suggested_label"),
        "type": anchor.get("suggested_type"),
        "reward": anchor.get("suggested_reward"),
        "source": anchor_source(anchor),
        "reason": None,
        "failure_reason": None,
    }, session_id)


def anchor_id_from_source(source):
    parts = str(source or "").split(":")
    return parts[1] if len(parts) >= 2 and parts[0] == "anchor" else None


def read_anchors(path):
    anchors = read_jsonl(path)
    if anchors.empty:
        print("Labeling started without anchors.")
    else:
        print("Found %d anchors at %s" % (len(anchors), path))
    return anchors


def load_label_state(path, session_id):
    if not path.exists():
        return {}, []
    with open(path, "r") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        if any(key in raw for key in ("labels", "auto_labels", "rejected_anchors")):
            rows = raw.get("labels", [])
            auto_rows = raw.get("auto_labels", [])
            rejected_rows = raw.get("rejected_anchors", [])
        else:
            rows = list(raw.values())
            auto_rows = []
            rejected_rows = []
    elif isinstance(raw, list):
        rows = raw
        auto_rows = []
        rejected_rows = []
    else:
        rows = []
        auto_rows = []
        rejected_rows = []

    labels = {}
    for row in list(rows) + list(auto_rows):
        if not isinstance(row, dict):
            continue
        label = normalize_label(row, session_id)
        label_id = label.get("id")
        if not label_id:
            label_id = next_auto_label_id(labels) if str(label.get("source") or "").startswith("anchor:") else next_label_id(labels)
        label["id"] = label_id
        labels[label_id] = label
    return labels, [row for row in rejected_rows if isinstance(row, dict)]


def save_labels(path, labels):
    rows = [labels[label_id] for label_id in sorted(labels.keys(), key=label_sort_key)]
    with open(path, "w") as f:
        json.dump({"labels": rows}, f, indent=2)
    print("Saved %d labels to %s" % (len(rows), path))


def load_labels_from_anchors(labels, anchors, session_id):
    if anchors.empty:
        return
    required = {"start_tick", "end_tick", "suggested_label"}
    missing = required.difference(set(anchors.columns))
    if missing:
        raise KeyError("Failed to load anchors. Missing fields: %s" % ", ".join(sorted(missing)))

    for _, anchor in anchors.iterrows():
        label_id = next_label_id(labels)
        source = anchor.get("source")
        labels[label_id] = normalize_label({
            "id": label_id,
            "session": anchor.get("session") or session_id,
            "start_tick": anchor.get("start_tick"),
            "end_tick": anchor.get("end_tick"),
            "label": anchor.get("suggested_label"),
            "type": anchor.get("suggested_type"),
            "reward": anchor.get("suggested_reward"),
            "source": "anchor" if pd.isna(source) or source is None else "anchor:%s" % source,
            "reason": None,
            "failure_reason": None,
        }, session_id)
    print("Loaded %d anchors as suggested labels." % len(anchors))


def split_anchors(anchors, session_id, labels, rejected_anchors):
    review_queue = []
    if anchors.empty:
        return review_queue

    accepted_anchor_ids = set()
    for label in labels.values():
        anchor_id = anchor_id_from_source(label.get("source"))
        if anchor_id:
            accepted_anchor_ids.add(anchor_id)
    rejected_anchor_ids = {str(anchor.get("id")) for anchor in rejected_anchors if anchor.get("id") is not None}

    for _, anchor_row in anchors.iterrows():
        anchor = {key: clean_value(value) for key, value in anchor_row.to_dict().items()}
        anchor_id = str(anchor.get("id")) if anchor.get("id") is not None else None
        if anchor_id is not None and (anchor_id in accepted_anchor_ids or anchor_id in rejected_anchor_ids):
            continue

        policy = anchor.get("review_policy") or "review"
        if policy == "ignore":
            continue
        if policy == "auto":
            label_id = next_auto_label_id(labels)
            labels[label_id] = label_from_anchor(anchor, session_id, label_id)
        else:
            review_queue.append(anchor)
    return review_queue


def navigation_menu():
    return """
    ---------------------------------
    Project Lipsy Recorder Label Tool
    ---------------------------------
    i - insert label
    e - edit label
    d - delete label
    m - merge labels
    review - review queued anchors
    v - view labels
    f - find labels
    g <tick> - set cursor tick
    n - move cursor to next label start_tick
    p - move cursor to previous label start_tick
    s - save labels
    r - return to prev menu
    q - save and quit
    """

def parse_instructions(cmd: str) -> list:
    try:
        instructions = shlex.split(cmd)
    except ValueError as exc:
        print("Invalid command: %s" % exc)
        return []
    if len(instructions) < 2:
        return []
    for i in range(len(instructions)):
        instructions[i] = clean_value(instructions[i])
    return instructions


def parse_field_updates(tokens):
    updates = {}
    i = 0
    while i < len(tokens):
        field = resolve_field_name(tokens[i])
        if field not in EDITABLE_FIELDS:
            print("Unknown field: %s" % tokens[i])
            i += 1
            continue
        if i + 1 >= len(tokens):
            print("Missing value for field: %s" % tokens[i])
            break

        if field in ("reason", "failure_reason"):
            value_parts = []
            j = i + 1
            while j < len(tokens) and not is_field_update_boundary(tokens, j):
                value_parts.append(str(tokens[j]))
                j += 1
            updates[field] = coerce_field_value(field, " ".join(value_parts) if value_parts else None)
            i = j
        else:
            updates[field] = coerce_field_value(field, tokens[i + 1])
            i += 2
    return updates


def is_field_update_boundary(tokens, index):
    if resolve_field_name(tokens[index]) is None:
        return False
    return index + 1 < len(tokens) and resolve_field_name(tokens[index + 1]) is None


def prompt_value(message, default=None, cast=str, allow_empty=False):
    suffix = " [%s]" % default if default is not None else ""
    value = input("%s%s: " % (message, suffix)).strip()
    if value.lower() == "r":
        return "r"
    if value == "" and default is not None:
        return default
    if value == "" and allow_empty:
        return None
    while value == "":
        value = input("%s%s: " % (message, suffix)).strip()
        if value.lower() == "r":
            return "r"
        if value == "" and default is not None:
            return default
        if value == "" and allow_empty:
            return None
    try:
        return cast(value)
    except ValueError:
        print("Invalid value. Try again or enter r to return.")
        return prompt_value(message, default=default, cast=cast, allow_empty=allow_empty)


def parse_reward(value):
    if value is None or value == "":
        return None
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip()
    if text.lower() in ("none", "null"):
        return None
    return float(text)


def coerce_field_value(field, value):
    if value is None:
        return None
    if field in ("start_tick", "end_tick"):
        return int(value)
    if field == "reward":
        return parse_reward(value)
    return value


def insert_label(labels, session_id, instructions=None):
    if instructions is None or instructions == []:
        print("Insert label. Enter r at any prompt to return.")
        start_tick = prompt_value("start_tick", cast=int)
        if start_tick == "r":
            return
        end_tick = prompt_value("end_tick", cast=int)
        if end_tick == "r":
            return
        label = prompt_value("label", cast=str)
        if label == "r":
            return
        label_type = prompt_value("type", cast=str, allow_empty=True)
        if label_type == "r":
            return
        reward = prompt_value("reward", cast=parse_reward, allow_empty=True)
        if reward == "r":
            return
        source = prompt_value("source", default="manual", cast=str)
        if source == "r":
            return
        reason = prompt_value("reason", cast=str, allow_empty=True)
        if reason == "r":
            return
        failure_reason = prompt_value("failure_reason", cast=str, allow_empty=True)
        if failure_reason == "r":
            return
    else:
        if len(instructions) <= 3:
            print("Invalid argument.")
            return
        start_tick, end_tick, label = int(instructions[1]), int(instructions[2]), instructions[3]
        label_type = instructions[4] if len(instructions) >= 5 else None
        reward = parse_reward(instructions[5]) if len(instructions) >= 6 else 0
        source = "manual"
        reason = None
        failure_reason = None
        if len(instructions) >= 7:
            field = resolve_field_name(instructions[6])
            if field in ("reason", "failure_reason"):
                text = " ".join(str(part) for part in instructions[7:]) if len(instructions) >= 8 else None
                if field == "reason":
                    reason = text
                else:
                    failure_reason = text
            else:
                reason = " ".join(str(part) for part in instructions[6:])

    label_id = next_label_id(labels)
    labels[label_id] = normalize_label({
        "id": label_id,
        "session": session_id,
        "start_tick": start_tick,
        "end_tick": end_tick,
        "label": label,
        "type": label_type,
        "reward": reward,
        "source": source,
        "reason": reason,
        "failure_reason": failure_reason,
    }, session_id)
    print("Inserted label %s." % label_id)


def view_labels(labels, instructions=None, cursor_tick=None):
    if not labels:
        print("No labels.")
        return
    if instructions is not None and len(instructions) >= 2:
        print_labels(labels, select_labels(labels, op=instructions[1]))
        return
    if cursor_tick is not None:
        label_ids = labels_around_cursor(labels, cursor_tick, radius=3)
        print("cursor_tick=%s" % cursor_tick)
        print_labels(labels, label_ids)
        return
    print(labels_to_dataframe(labels, drop_session=True).to_string(index=False))


def print_labels(labels, label_ids):
    selected = {label_id: labels[label_id] for label_id in label_ids if label_id in labels}
    if not selected:
        print("No matching labels.")
        return
    print(labels_to_dataframe(selected, drop_session=True).to_string(index=False))


def find_labels(labels, instructions=None):
    op = instructions[1] if instructions is not None and len(instructions) >= 2 else None
    label_ids = select_labels(labels, op=op)
    print_labels(labels, label_ids)
    return label_ids


def select_labels(labels, op=None):
    if not labels:
        print("No labels.")
        return []

    if op is None:
        print("""
        Find labels by:
        id - #exact label id
        ids - comma-separated ids
        tick - @labels containing tick
        range - labels overlapping start-end
        label - label text
        type - type text
        all - all labels
        r - return
        """)
        op = input("> ").strip().lower()
    if not op:
        return []
    if op == "r":
        return []
    if op == "all":
        return sorted(labels.keys(), key=label_sort_key)
    if op == "id":
        label_id = prompt_value("label id", cast=str)
        if label_id == "r":
            return []
        resolved = resolve_label_id(labels, label_id)
        return [resolved] if resolved is not None else []
    if op[0] == "#":
        label_id = op[1:].strip()
        resolved = resolve_label_id(labels, label_id)
        return [resolved] if resolved is not None else []
    if op == "ids":
        value = prompt_value("label ids, comma-separated", cast=str)
        if value == "r":
            return []
        label_ids = []
        for item in value.split(","):
            resolved = resolve_label_id(labels, item)
            if resolved is not None:
                label_ids.append(resolved)
        return label_ids
    if "," in op and op[0] != "#":
        label_ids = []
        for item in op.split(","):
            resolved = resolve_label_id(labels, item)
            if resolved is not None:
                label_ids.append(resolved)
        return label_ids
    if op == "tick":
        tick = prompt_value("tick", cast=int)
        if tick == "r":
            return []
        return [
            label_id for label_id, label in labels.items()
            if label.get("start_tick") is not None
            and label.get("end_tick") is not None
            and int(label["start_tick"]) <= tick <= int(label["end_tick"])
        ]
    if op[0] == "@":
        value = op[1:].strip()
        if not value.isdigit():
            return []
        tick = int(value)
        return [
            label_id for label_id, label in labels.items()
            if label.get("start_tick") is not None
               and label.get("end_tick") is not None
               and int(label["start_tick"]) <= tick <= int(label["end_tick"])
        ]
    if op == "range":
        start_tick = prompt_value("range start_tick", cast=int)
        if start_tick == "r":
            return []
        end_tick = prompt_value("range end_tick", cast=int)
        if end_tick == "r":
            return []
        return [
            label_id for label_id, label in labels.items()
            if label.get("start_tick") is not None
            and label.get("end_tick") is not None
            and int(label["start_tick"]) <= end_tick
            and int(label["end_tick"]) >= start_tick
        ]
    if "-" in op and op[0] != "@" and op[0] != "#":
        start_tick, end_tick = op.split("-")
        start_tick = int(start_tick) if start_tick.isdigit() else -1
        end_tick = int(end_tick) if end_tick.isdigit() else -1
        return [
            label_id for label_id, label in labels.items()
            if label.get("start_tick") is not None
               and label.get("end_tick") is not None
               and int(label["start_tick"]) <= end_tick
               and int(label["end_tick"]) >= start_tick
        ]
    find_field = resolve_field_name(op)
    if find_field in ("label", "type", "source", "reason", "failure_reason"):
        value = prompt_value("%s contains" % find_field, cast=str)
        if value == "r":
            return []
        value = value.lower()
        return [
            label_id for label_id, label in labels.items()
            if value in str(label.get(find_field) or "").lower()
        ]
    if not any(char.isdigit() for char in op):
        if op == "r":
            return []
        return [
            label_id for label_id, label in labels.items()
            if op in str(label.get("label") or "").lower() or
            op in str(label.get("type") or "").lower() or
            op in str(label.get("source") or "").lower() or
            op in str(label.get("reason") or "").lower() or
            op in str(label.get("failure_reason") or "").lower()
        ]
    print("Unknown find option.")


def edit_labels(labels, session_id, instructions=None, current_tick=None):
    if instructions is None or instructions == []:
        label_ids = select_labels(labels, op="@%s" % current_tick) if current_tick is not None else []
        if not label_ids:
            label_ids = select_labels(labels)
        if not label_ids:
            return
        print_labels(labels, label_ids)

        while True:
            print("Editable fields: %s" % ", ".join(EDITABLE_FIELDS))
            field = input("field to edit, or r to return: ").strip()
            if field.lower() == "r":
                return
            field = resolve_field_name(field)
            if field not in EDITABLE_FIELDS:
                print("Unknown field.")
                continue

            current = labels[label_ids[0]].get(field) if len(label_ids) == 1 else None
            cast = int if field in ("start_tick", "end_tick") else parse_reward if field == "reward" else str
            value = prompt_value("new %s" % field, default=current, cast=cast, allow_empty=True)
            if value == "r":
                return

            for label_id in label_ids:
                labels[label_id][field] = value
                labels[label_id]["session"] = labels[label_id].get("session") or session_id
            print("Updated %d label(s)." % len(label_ids))
            print_labels(labels, label_ids)
    else:
        selector = instructions[1]
        label_ids = select_labels(labels, op=selector)
        if not label_ids:
            return

        updates = parse_field_updates(instructions[2:])
        if not updates:
            return
        for field, value in updates.items():
            for label_id in label_ids:
                labels[label_id][field] = value
                labels[label_id]["session"] = labels[label_id].get("session") or session_id
        print("Updated %d label(s)." % len(label_ids))
        print_labels(labels, label_ids)


def delete_labels(labels, instructions=None, current_tick=None):
    if instructions is None or instructions == []:
        label_ids = select_labels(labels, op="@%s" % current_tick) if current_tick is not None else []
        if not label_ids:
            label_ids = select_labels(labels)
    else:
        label_ids = select_labels(labels, op=instructions[1])
    if not label_ids:
        return
    print_labels(labels, label_ids)
    confirm = input("Delete these label(s)? y/N/r: ").strip().lower()
    if confirm in ("r", "", "n", "no"):
        return
    if confirm not in ("y", "yes"):
        print("Delete canceled.")
        return
    for label_id in label_ids:
        labels.pop(label_id, None)
    print("Deleted %d label(s)." % len(label_ids))


def merge_text_values(values):
    parts = []
    for value in values:
        value = clean_value(value)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            parts.append(text)
    return " ".join(parts) if parts else None


def choose_label_value(values):
    unique_values = []
    for value in values:
        value = clean_value(value)
        if value is None:
            continue
        if value not in unique_values:
            unique_values.append(value)

    if not unique_values:
        return None
    if len(unique_values) == 1:
        return unique_values[0]

    while True:
        print("Labels differ. Choose label to keep:")
        for idx, value in enumerate(unique_values, start=1):
            print("%d - %s" % (idx, value))
        value = input("label choice, or r to return: ").strip().lower()
        if value == "r":
            return "r"
        if value.isdigit() and 1 <= int(value) <= len(unique_values):
            return unique_values[int(value) - 1]
        print("Invalid choice.")


def merge_labels(labels, session_id, instructions=None):
    if instructions is None or instructions == []:
        label_ids = select_labels(labels)
        updates = {}
    else:
        if len(instructions) < 2:
            print("Missing merge selector.")
            return
        label_ids = select_labels(labels, op=instructions[1])
        updates = parse_field_updates(instructions[2:]) if len(instructions) > 2 else {}
    label_ids = unique_label_ids(label_ids)
    if len(label_ids) < 2:
        print("Select at least two labels to merge.")
        return

    label_ids = sorted(label_ids, key=label_sort_key)
    print_labels(labels, label_ids)
    selected = [labels[label_id] for label_id in label_ids]
    merged_label = choose_label_value([label.get("label") for label in selected])
    if merged_label == "r":
        return

    start_ticks = [int(label["start_tick"]) for label in selected if label.get("start_tick") is not None]
    end_ticks = [int(label["end_tick"]) for label in selected if label.get("end_tick") is not None]
    if not start_ticks or not end_ticks:
        print("Cannot merge labels with missing start_tick/end_tick.")
        return

    keep_id = label_ids[0]
    source = merge_text_values(label.get("source") for label in selected)
    merged = normalize_label({
        "id": keep_id,
        "session": selected[0].get("session") or session_id,
        "start_tick": min(start_ticks),
        "end_tick": max(end_ticks),
        "label": merged_label,
        "type": merge_text_values(label.get("type") for label in selected),
        "reward": merge_text_values(label.get("reward") for label in selected),
        "source": "merged: " + source if source is not None else None,
        "reason": merge_text_values(label.get("reason") for label in selected),
        "failure_reason": merge_text_values(label.get("failure_reason") for label in selected),
    }, session_id)
    for field, value in updates.items():
        merged[field] = value

    for label_id in label_ids:
        labels.pop(label_id, None)
    labels[keep_id] = merged
    print("Merged %d labels into %s." % (len(label_ids), keep_id))
    print_labels(labels, [keep_id])


def sorted_label_ids_by_start(labels):
    return sorted(
        labels.keys(),
        key=lambda label_id: (
            int(labels[label_id].get("start_tick")) if labels[label_id].get("start_tick") is not None else 10 ** 12,
            label_sort_key(label_id),
        )
    )


def labels_around_cursor(labels, cursor_tick, radius=3):
    sorted_ids = sorted_label_ids_by_start(labels)
    if not sorted_ids:
        return []

    active_ids = [
        label_id for label_id in sorted_ids
        if labels[label_id].get("start_tick") is not None
        and labels[label_id].get("end_tick") is not None
        and int(labels[label_id]["start_tick"]) <= cursor_tick <= int(labels[label_id]["end_tick"])
    ]

    insert_at = len(sorted_ids)
    for idx, label_id in enumerate(sorted_ids):
        start_tick = labels[label_id].get("start_tick")
        if start_tick is not None and int(start_tick) >= cursor_tick:
            insert_at = idx
            break

    start_idx = max(0, insert_at - radius)
    end_idx = min(len(sorted_ids), insert_at + radius + 1)
    neighborhood_ids = sorted_ids[start_idx:end_idx]
    selected_ids = set(active_ids + neighborhood_ids)
    return [label_id for label_id in sorted_ids if label_id in selected_ids]


def cursor_goto(instructions, current_tick=None):
    if instructions is None or len(instructions) < 2:
        value = prompt_value("cursor tick", default=current_tick, cast=int)
        if value == "r":
            return current_tick
        return value
    try:
        return int(instructions[1])
    except (TypeError, ValueError):
        print("Invalid cursor tick.")
        return current_tick


def cursor_next_label(labels, cursor_tick):
    sorted_ids = sorted_label_ids_by_start(labels)
    for label_id in sorted_ids:
        start_tick = labels[label_id].get("start_tick")
        if start_tick is not None and int(start_tick) > cursor_tick:
            return int(start_tick)
    print("No next label.")
    return cursor_tick


def cursor_previous_label(labels, cursor_tick):
    sorted_ids = sorted_label_ids_by_start(labels)
    previous_tick = None
    for label_id in sorted_ids:
        start_tick = labels[label_id].get("start_tick")
        if start_tick is None:
            continue
        start_tick = int(start_tick)
        if start_tick >= cursor_tick:
            break
        previous_tick = start_tick
    if previous_tick is None:
        print("No previous label.")
        return cursor_tick
    return previous_tick


def print_summary(total_anchors, review_queue, rejected_anchors, labels):
    auto_count = sum(1 for label_id in labels.keys() if str(label_id).startswith("auto_"))
    manual_count = len(labels) - auto_count
    print("Summary:")
    print("  total anchors: %d" % total_anchors)
    print("  auto accepted: %d" % auto_count)
    print("  review required: %d" % len(review_queue))
    print("  rejected: %d" % len(rejected_anchors))
    print("  final manual labels: %d" % manual_count)
    print("  total labels: %d" % len(labels))


def print_review_item(anchor, index, total, status=None):
    print("\nReview %d/%d" % (index + 1, total))
    if status is not None:
        print("status: %s" % status)
    preferred_fields = [
        "id",
        "review_policy",
        "session",
        "start_tick",
        "end_tick",
        "suggested_label",
        "suggested_type",
        "suggested_reward",
        "source",
    ]
    row = {}
    for field in preferred_fields:
        if field in anchor:
            row[field] = clean_value(anchor.get(field))
    for field, value in anchor.items():
        if field not in row:
            row[field] = clean_value(value)
    print(pd.DataFrame([row]).to_string(index=False))
    label = anchor.get("suggested_label")
    presets = REVIEW_PRESETS.get(label, {})
    if presets:
        print("Presets:")
        for key, (label_type, reward) in sorted(presets.items()):
            print("  %s: %s %+g" % (key, label_type, reward))
    print("Enter=accept, x=reject, number=preset, e=manual edit, n/p=next/prev, r=return")


def find_review_label_id(labels, anchor):
    anchor_id = clean_value(anchor.get("id"))
    expected_source = anchor_source(anchor)
    for label_id, label in labels.items():
        source = label.get("source")
        if source == expected_source:
            return label_id
        if anchor_id is not None and anchor_id_from_source(source) == str(anchor_id):
            return label_id
    return None


def remove_rejected_anchor(rejected_anchors, anchor):
    target_key = review_anchor_key(anchor)
    for idx in range(len(rejected_anchors) - 1, -1, -1):
        if review_anchor_key(rejected_anchors[idx]) == target_key:
            rejected_anchors.pop(idx)


def review_anchor_status(labels, rejected_anchors, pending_keys, anchor):
    anchor_key = review_anchor_key(anchor)
    if find_review_label_id(labels, anchor) is not None:
        return "accepted"
    if any(review_anchor_key(rejected_anchor) == anchor_key for rejected_anchor in rejected_anchors):
        return "rejected"
    if anchor_key in pending_keys:
        return "pending"
    return "reviewed"


def accept_review_anchor(labels, session_id, anchor, rejected_anchors, label_type=None, reward=None):
    label_id = find_review_label_id(labels, anchor) or next_label_id(labels)
    label = label_from_anchor(anchor, session_id, label_id)
    if label_type is not None:
        label["type"] = label_type
    if reward is not None:
        label["reward"] = reward
    labels[label_id] = label
    remove_rejected_anchor(rejected_anchors, anchor)
    return label_id


def manual_review_label(labels, session_id, anchor, rejected_anchors):
    existing_id = find_review_label_id(labels, anchor)
    existing = labels.get(existing_id, {}) if existing_id is not None else {}
    print("Manual review edit. Empty keeps suggested value; r returns.")
    start_tick = prompt_value("start_tick", default=existing.get("start_tick"), cast=int)
    if start_tick == "r":
        return None
    end_tick = prompt_value("end_tick", default=existing.get("end_tick"), cast=int)
    if end_tick == "r":
        return None
    label = prompt_value("label", default=existing.get("label") or anchor.get("suggested_label"), cast=str)
    if label == "r":
        return None
    label_type = prompt_value("type", default=existing.get("type") or anchor.get("suggested_type"), cast=str, allow_empty=True)
    if label_type == "r":
        return None
    reward = prompt_value("reward", default=existing.get("reward") if existing.get("reward") is not None else anchor.get("suggested_reward"), cast=parse_reward, allow_empty=True)
    if reward == "r":
        return None
    reason = prompt_value("reason", default=existing.get("reason"), cast=str, allow_empty=True)
    if reason == "r":
        return None
    failure_reason = prompt_value("failure_reason", default=existing.get("failure_reason"), cast=str, allow_empty=True)
    if failure_reason == "r":
        return None

    label_id = existing_id or next_label_id(labels)
    labels[label_id] = normalize_label({
        "id": label_id,
        "session": anchor.get("session") or session_id,
        "start_tick": anchor.get("start_tick"),
        "end_tick": anchor.get("end_tick"),
        "label": label,
        "type": label_type,
        "reward": reward,
        "source": anchor_source(anchor),
        "reason": reason,
        "failure_reason": failure_reason,
    }, session_id)
    remove_rejected_anchor(rejected_anchors, anchor)
    return label_id


def review_anchor_key(anchor):
    anchor_id = clean_value(anchor.get("id"))
    if anchor_id is not None:
        return "id:%s" % anchor_id
    return "span:%s:%s:%s:%s" % (
        clean_value(anchor.get("session")),
        clean_value(anchor.get("start_tick")),
        clean_value(anchor.get("end_tick")),
        clean_value(anchor.get("source")),
    )


def remove_review_anchor(review_queue, anchor):
    target_key = review_anchor_key(anchor)
    for idx, queued_anchor in enumerate(review_queue):
        if review_anchor_key(queued_anchor) == target_key:
            review_queue.pop(idx)
            return


def next_pending_review_index(review_items, pending_keys, start_index):
    if not pending_keys:
        return None
    for idx in range(start_index, len(review_items)):
        if review_anchor_key(review_items[idx]) in pending_keys:
            return idx
    for idx in range(0, min(start_index, len(review_items))):
        if review_anchor_key(review_items[idx]) in pending_keys:
            return idx
    return None


def review_mode(labels, session_id, review_queue, rejected_anchors):
    if not review_queue:
        print("No review anchors.")
        return

    index = 0
    review_items = list(review_queue)
    pending_keys = {review_anchor_key(anchor) for anchor in review_queue}
    while review_items:
        index = max(0, min(index, len(review_items) - 1))
        anchor = review_items[index]
        anchor_key = review_anchor_key(anchor)
        print_review_item(anchor, index, len(review_items), review_anchor_status(labels, rejected_anchors, pending_keys, anchor))
        op = input("review> ").strip().lower()

        if op == "r":
            return
        if op == "p":
            index = max(0, index - 1)
            continue
        if op == "n":
            index = min(len(review_items) - 1, index + 1)
            continue
        if op == "x":
            existing_id = find_review_label_id(labels, anchor)
            if existing_id is not None:
                labels.pop(existing_id, None)
            remove_rejected_anchor(rejected_anchors, anchor)
            rejected_anchors.append(anchor)
            pending_keys.discard(anchor_key)
            remove_review_anchor(review_queue, anchor)
            next_index = next_pending_review_index(review_items, pending_keys, index + 1)
            if next_index is not None:
                index = next_index
            continue
        if op == "e":
            label_id = manual_review_label(labels, session_id, anchor, rejected_anchors)
            if label_id is not None:
                print("Accepted review anchor as %s." % label_id)
                pending_keys.discard(anchor_key)
                remove_review_anchor(review_queue, anchor)
                next_index = next_pending_review_index(review_items, pending_keys, index + 1)
                if next_index is not None:
                    index = next_index
            continue

        label = anchor.get("suggested_label")
        presets = REVIEW_PRESETS.get(label, {})
        if op in presets:
            label_type, reward = presets[op]
            label_id = accept_review_anchor(labels, session_id, anchor, rejected_anchors, label_type, reward)
            print("Accepted review anchor as %s." % label_id)
            pending_keys.discard(anchor_key)
            remove_review_anchor(review_queue, anchor)
            next_index = next_pending_review_index(review_items, pending_keys, index + 1)
            if next_index is not None:
                index = next_index
            continue
        if op == "":
            label_id = accept_review_anchor(labels, session_id, anchor, rejected_anchors)
            print("Accepted review anchor as %s." % label_id)
            pending_keys.discard(anchor_key)
            remove_review_anchor(review_queue, anchor)
            next_index = next_pending_review_index(review_items, pending_keys, index + 1)
            if next_index is not None:
                index = next_index
            continue
        print("Unknown review command.")

def main():
    session_dir = Path(input("Enter folder directory: ")).expanduser()
    meta_path = session_dir / "session_meta.json"
    events_path = session_dir / "events.jsonl"
    frames_path = find_frames_path(session_dir)
    anchors_path = session_dir / "anchors.jsonl"
    labels_path = session_dir / "labels.json"

    with open(meta_path, "r") as f:
        meta = json.load(f)

    read_jsonl(events_path)
    read_jsonl(frames_path)

    session_id = meta.get("sessionId") or meta.get("session_id") or session_dir.name
    labels, rejected_anchors = load_label_state(labels_path, session_id)
    anchors = read_anchors(anchors_path)
    review_queue = split_anchors(anchors, session_id, labels, rejected_anchors)
    total_anchors = len(anchors) if not anchors.empty else 0
    if labels:
        print("Loaded %d labels from %s." % (len(labels), labels_path))
    print_summary(total_anchors, review_queue, rejected_anchors, labels)

    current_tick = 0
    print(navigation_menu())
    while True:
        cmd = input("> ").strip().lower()
        if cmd == "":
            continue
        instructions = parse_instructions(cmd)
        op = cmd[0]
        if cmd == "review":
            review_mode(labels, session_id, review_queue, rejected_anchors)
            print_summary(total_anchors, review_queue, rejected_anchors, labels)
        elif op == "r":
            print(navigation_menu())
        elif op == "i":
            insert_label(labels, session_id, instructions)
        elif op == "e":
            edit_labels(labels, session_id, instructions, current_tick)
        elif op == "d":
            delete_labels(labels, instructions, current_tick)
        elif op == "m":
            merge_labels(labels, session_id, instructions)
        elif op == "v":
            view_labels(labels, instructions, current_tick)
        elif op == "f":
            find_labels(labels, instructions)
        elif op == "g":
            current_tick = cursor_goto(instructions, current_tick)
            print("cursor_tick=%s" % current_tick)
        elif op == "n":
            current_tick = cursor_next_label(labels, current_tick)
            print("cursor_tick=%s" % current_tick)
            view_labels(labels, [], current_tick)
        elif op == "p":
            current_tick = cursor_previous_label(labels, current_tick)
            print("cursor_tick=%s" % current_tick)
            view_labels(labels, [], current_tick)
        elif op == "s":
            save_labels(labels_path, labels)
            print_summary(total_anchors, review_queue, rejected_anchors, labels)
        elif op == "q":
            save_labels(labels_path, labels)
            print_summary(total_anchors, review_queue, rejected_anchors, labels)
            return
        elif op == "h" or op == "help":
            print(navigation_menu())


if __name__ == "__main__":
    main()
