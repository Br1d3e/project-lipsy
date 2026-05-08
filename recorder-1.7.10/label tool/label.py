import json
import math
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
    "notes",
]

EDITABLE_FIELDS = [
    "start_tick",
    "end_tick",
    "label",
    "type",
    "reward",
    "source",
    "notes",
]


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


def normalize_label(label, session_id):
    normalized = {field: clean_value(label.get(field)) for field in LABEL_FIELDS}
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


def next_label_id(labels):
    max_id = 0
    for label_id in labels.keys():
        digits = "".join(ch for ch in str(label_id) if ch.isdigit())
        if digits:
            max_id = max(max_id, int(digits))
    return "lbl_%06d" % (max_id + 1)


def read_anchors(path):
    anchors = read_jsonl(path)
    if anchors.empty:
        print("Labeling started without anchors.")
    else:
        print("Found %d anchors at %s" % (len(anchors), path))
    return anchors


def load_labels(path, session_id):
    if not path.exists():
        return {}
    with open(path, "r") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        if "labels" in raw and isinstance(raw["labels"], list):
            rows = raw["labels"]
        else:
            rows = list(raw.values())
    elif isinstance(raw, list):
        rows = raw
    else:
        rows = []

    labels = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = normalize_label(row, session_id)
        label_id = label.get("id") or next_label_id(labels)
        label["id"] = label_id
        labels[label_id] = label
    return labels


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
            "notes": None,
        }, session_id)
    print("Loaded %d anchors as suggested labels." % len(anchors))


def navigation_menu():
    return """
    ---------------------------------
    Project Lipsy Recorder Label Tool
    ---------------------------------
    i - insert label
    e - edit label
    d - delete label
    m - merge labels
    v - view labels
    f - find labels
    s - save labels
    r - return to prev menu
    q - save and quit
    """


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


def insert_label(labels, session_id):
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
    notes = prompt_value("notes", cast=str, allow_empty=True)
    if notes == "r":
        return

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
        "notes": notes,
    }, session_id)
    print("Inserted label %s." % label_id)


def view_labels(labels):
    if not labels:
        print("No labels.")
        return
    print(labels_to_dataframe(labels, drop_session=True).to_string(index=False))


def print_labels(labels, label_ids):
    selected = {label_id: labels[label_id] for label_id in label_ids if label_id in labels}
    if not selected:
        print("No matching labels.")
        return
    print(labels_to_dataframe(selected, drop_session=True).to_string(index=False))


def find_labels(labels):
    label_ids = select_labels(labels)
    print_labels(labels, label_ids)
    return label_ids


def select_labels(labels):
    if not labels:
        print("No labels.")
        return []

    while True:
        print("""
        Find labels by:
        id - exact label id
        ids - comma-separated ids
        tick - labels containing tick
        range - labels overlapping start/end
        label - label text
        type - type text
        all - all labels
        r - return
        """)
        op = input("find> ").strip().lower()
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
        if op in ("label", "type"):
            value = prompt_value("%s contains" % op, cast=str)
            if value == "r":
                return []
            value = value.lower()
            return [
                label_id for label_id, label in labels.items()
                if value in str(label.get(op) or "").lower()
            ]
        print("Unknown find option.")


def edit_labels(labels, session_id):
    label_ids = select_labels(labels)
    if not label_ids:
        return
    print_labels(labels, label_ids)

    while True:
        print("Editable fields: %s" % ", ".join(EDITABLE_FIELDS))
        field = input("field to edit, or r to return: ").strip()
        if field.lower() == "r":
            return
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


def delete_labels(labels):
    label_ids = select_labels(labels)
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


def merge_labels(labels, session_id):
    label_ids = select_labels(labels)
    if len(label_ids) < 2:
        print("Select at least two labels to merge.")
        return

    label_ids = sorted(set(label_ids), key=label_sort_key)
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
        "notes": merge_text_values(label.get("notes") for label in selected),
    }, session_id)

    for label_id in label_ids:
        labels.pop(label_id, None)
    labels[keep_id] = merged
    print("Merged %d labels into %s." % (len(label_ids), keep_id))
    print_labels(labels, [keep_id])


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
    labels = load_labels(labels_path, session_id)
    if labels:
        print("Loaded %d labels from %s." % (len(labels), labels_path))
    else:
        anchors = read_anchors(anchors_path)
        load_labels_from_anchors(labels, anchors, session_id)

    print(navigation_menu())
    while True:
        op = input("> ").strip().lower()
        if op == "r":
            print(navigation_menu())
        elif op == "i":
            insert_label(labels, session_id)
        elif op == "e":
            edit_labels(labels, session_id)
        elif op == "d":
            delete_labels(labels)
        elif op == "m":
            merge_labels(labels, session_id)
        elif op == "v":
            view_labels(labels)
        elif op == "f":
            find_labels(labels)
        elif op == "s":
            save_labels(labels_path, labels)
        elif op == "q":
            save_labels(labels_path, labels)
            return
        elif op == "h" or op == "help":
            print(navigation_menu())


if __name__ == "__main__":
    main()
