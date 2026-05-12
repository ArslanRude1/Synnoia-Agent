from __future__ import annotations

import json
import re
import sys
from typing import Any

CITATION_RE = re.compile(r"^\[\d+\]$")


def _omit_nulls(d: dict) -> dict:
    """Return a copy of *d* with all None / empty-dict values removed."""
    return {k: v for k, v in d.items() if v is not None and v != {}}


def _is_citation(text: str) -> bool:
    return bool(CITATION_RE.match(text.strip()))


def load_tiptap(filepath: str) -> dict:
    """Load raw TipTap JSON from *filepath*."""
    with open(filepath, encoding="utf-8") as fh:
        return json.load(fh)


def _marks_to_formatting(marks: list[dict]) -> dict:
    """
    Convert a list of TipTap mark objects into a flat formatting dict.
    Only includes non-None attributes.
    """
    fmt: dict[str, Any] = {}
    for mark in marks:
        mtype = mark.get("type") or ""
        attrs = mark.get("attrs") or {}

        if mtype == "bold":
            fmt["bold"] = True
        elif mtype == "italic":
            fmt["italic"] = True
        elif mtype == "underline":
            fmt["underline"] = True
        elif mtype == "strike":
            fmt["strikethrough"] = True
        elif mtype == "superscript":
            fmt["superscript"] = True
        elif mtype == "subscript":
            fmt["subscript"] = True
        elif mtype == "highlight":
            color = attrs.get("color")
            if color:
                fmt["highlight"] = color
        elif mtype == "link":
            pass  # Links are intentionally omitted from Simple JSON (too verbose)
        elif mtype == "textStyle":
            family = attrs.get("fontFamily")
            size = attrs.get("fontSize")
            color = attrs.get("color")
            if family:
                fmt["fontFamily"] = family
            if size:
                fmt["fontSize"] = size
            if color:
                fmt["color"] = color

    return fmt


def extract_text_segments(content: list[dict]) -> list[dict]:
    """
    Convert a list of TipTap text nodes into clean segments.
    - Merges consecutive plain-text nodes (no marks).
    - Tags citation segments with ``is_citation: true``.
    - Never produces a segment whose ``formatting`` is empty.
    """
    segments: list[dict] = []

    for node in content:
        if node.get("type") != "text":
            continue

        text = node.get("text") or ""
        marks = node.get("marks") or []
        fmt = _marks_to_formatting(marks)

        seg: dict[str, Any] = {"text": text}
        if fmt:
            seg["formatting"] = fmt

        # Citation detection: superscript + link whose text matches [N]
        if fmt.get("superscript") and _is_citation(text):
            seg["is_citation"] = True

        # Merge with previous plain-text segment when both have no formatting
        if not fmt and segments and "formatting" not in segments[-1] and not segments[-1].get("is_citation"):
            segments[-1]["text"] += text
        else:
            segments.append(seg)

    return segments


def parse_heading(node: dict) -> dict:
    attrs = node.get("attrs") or {}
    content = node.get("content") or []

    segments = extract_text_segments(content)
    # Heading text: join all segments for simplicity; keep segments for fidelity
    text = "".join(s["text"] for s in segments)

    fmt: dict[str, Any] = {}
    for s in segments:
        fmt.update(s.get("formatting") or {})

    result: dict[str, Any] = {
        "type": "heading",
        "level": attrs.get("level") or 2,
    }
    node_id = attrs.get("id")
    if node_id:
        result["id"] = node_id
    toc_id = attrs.get("data-toc-id")
    if toc_id:
        result["toc_id"] = toc_id
    result["text"] = text
    if fmt:
        result["formatting"] = fmt

    # Extra attrs
    for key in ("textAlign", "lineHeight", "indent", "margin"):
        v = attrs.get(key)
        if v is not None and v != {}:
            result[key] = v

    return result


def parse_paragraph(node: dict) -> dict:
    attrs = node.get("attrs") or {}
    content = node.get("content") or []

    result: dict[str, Any] = {"type": "paragraph"}

    # Extract node id
    node_id = attrs.get("id")
    if node_id:
        result["id"] = node_id

    for key, default in [("textAlign", None), ("lineHeight", None),
                          ("indent", None), ("margin", {})]:
        v = attrs.get(key)
        if v is None:
            v = default
        if v is not None and v != {}:
            result[key] = v

    segments = extract_text_segments(content)
    if segments:
        result["segments"] = segments

    return result


def parse_blockquote(node: dict) -> dict:
    attrs = node.get("attrs") or {}
    content = node.get("content") or []
    segments: list[dict] = []
    for child in content:
        if child.get("type") == "paragraph":
            segments.extend(extract_text_segments(child.get("content") or []))

    result: dict[str, Any] = {"type": "blockquote"}

    # Extract node id
    node_id = attrs.get("id")
    if node_id:
        result["id"] = node_id

    if segments:
        result["segments"] = segments
    return result


def _parse_list_item(item: dict) -> list[dict]:
    """
    Extract segments from all paragraph children of a listItem / taskItem.
    Returns a flat list of segments (multiple paragraphs → merged).
    """
    segments: list[dict] = []
    for child in item.get("content") or []:
        if child.get("type") == "paragraph":
            segments.extend(extract_text_segments(child.get("content") or []))
    return segments


def parse_ordered_list(node: dict) -> dict:
    attrs = node.get("attrs") or {}
    items_raw = node.get("content") or []

    list_items: list[dict] = []
    for idx, item in enumerate(items_raw, start=1):
        segs = _parse_list_item(item)
        entry: dict[str, Any] = {"index": idx}
        if segs:
            entry["segments"] = segs
        # Extract listItem id from attrs
        item_attrs = item.get("attrs") or {}
        item_id = item_attrs.get("id")
        if item_id:
            entry["id"] = item_id
        list_items.append(entry)

    result: dict[str, Any] = {
        "type": "ordered_list",
        "listType": attrs.get("listType") or "decimal",
    }

    # Extract node id
    node_id = attrs.get("id")
    if node_id:
        result["id"] = node_id

    if list_items:
        result["items"] = list_items
    return result


def parse_bullet_list(node: dict) -> dict:
    attrs = node.get("attrs") or {}
    items_raw = node.get("content") or []

    list_items: list[dict] = []
    for item in items_raw:
        segs = _parse_list_item(item)
        entry: dict[str, Any] = {}
        if segs:
            entry["segments"] = segs
        # Extract listItem id from attrs
        item_attrs = item.get("attrs") or {}
        item_id = item_attrs.get("id")
        if item_id:
            entry["id"] = item_id
        list_items.append(entry)

    result: dict[str, Any] = {
        "type": "bullet_list",
        "listType": attrs.get("listType") or "disc",
    }

    # Extract node id
    node_id = attrs.get("id")
    if node_id:
        result["id"] = node_id

    if list_items:
        result["items"] = list_items
    return result


def parse_task_list(node: dict) -> dict:
    attrs = node.get("attrs") or {}
    items_raw = node.get("content") or []

    list_items: list[dict] = []
    for item in items_raw:
        item_attrs = item.get("attrs") or {}
        checked = item_attrs.get("checked") or False
        segs = _parse_list_item(item)
        # Simple flat text for task items (usually single text node)
        text = "".join(s["text"] for s in segs)
        entry: dict[str, Any] = {"checked": checked, "text": text}
        # Extract taskItem id
        item_id = item_attrs.get("id")
        if item_id:
            entry["id"] = item_id
        list_items.append(entry)

    result: dict[str, Any] = {"type": "task_list"}

    # Extract node id
    node_id = attrs.get("id")
    if node_id:
        result["id"] = node_id

    if list_items:
        result["items"] = list_items
    return result


def parse_code_block(node: dict) -> dict:
    attrs = node.get("attrs") or {}
    content = node.get("content") or []
    code = "".join(n.get("text") or "" for n in content if n.get("type") == "text")

    result: dict[str, Any] = {
        "type": "code_block",
        "language": attrs.get("language") or "plaintext",
    }

    # Extract node id
    node_id = attrs.get("id")
    if node_id:
        result["id"] = node_id

    theme = attrs.get("theme")
    if theme:
        result["theme"] = theme
    word_wrap = attrs.get("wordWrap")
    if word_wrap is not None:
        result["wordWrap"] = word_wrap
    result["code"] = code
    return result


def _parse_table_cell(cell: dict) -> dict:
    """Parse a tableHeader or tableCell node into a simple cell dict."""
    attrs = cell.get("attrs") or {}
    content = cell.get("content") or []

    # Gather all text segments from inner paragraphs
    segments: list[dict] = []
    for child in content:
        if child.get("type") == "paragraph":
            segments.extend(extract_text_segments(child.get("content") or []))

    simple_text = "".join(s["text"] for s in segments)

    cell_obj: dict[str, Any] = {"text": simple_text}

    # Extract cell id
    cell_id = attrs.get("id")
    if cell_id:
        cell_obj["id"] = cell_id

    # Rich segments only if there is actual formatting
    has_fmt = any("formatting" in s for s in segments)
    if has_fmt:
        cell_obj["segments"] = segments

    # Preserve span info and styling if non-default
    colspan = attrs.get("colspan") or 1
    rowspan = attrs.get("rowspan") or 1
    if colspan and colspan != 1:
        cell_obj["colspan"] = colspan
    if rowspan and rowspan != 1:
        cell_obj["rowspan"] = rowspan
    for key in ("align", "background", "color", "colwidth"):
        v = attrs.get(key)
        if v is not None:
            cell_obj[key] = v

    return cell_obj


def parse_table(node: dict) -> dict:
    attrs = node.get("attrs") or {}
    rows_raw = node.get("content") or []
    rows: list[dict] = []

    for row_node in rows_raw:
        row_attrs = row_node.get("attrs") or {}
        cells_raw = row_node.get("content") or []
        row_type = "header" if any(c.get("type") == "tableHeader" for c in cells_raw) else "body"
        cells = [_parse_table_cell(c) for c in cells_raw]
        row_obj: dict[str, Any] = {"row_type": row_type, "cells": cells}
        # Extract row id
        row_id = row_attrs.get("id")
        if row_id:
            row_obj["id"] = row_id
        rows.append(row_obj)

    result: dict[str, Any] = {"type": "table", "rows": rows}

    # Extract node id
    node_id = attrs.get("id")
    if node_id:
        result["id"] = node_id

    return result


def parse_node(node: dict) -> dict | None:
    """Dispatch a single TipTap node to the right parser."""
    ntype = node.get("type") or ""
    dispatch = {
        "heading": parse_heading,
        "paragraph": parse_paragraph,
        "blockquote": parse_blockquote,
        "orderedList": parse_ordered_list,
        "bulletList": parse_bullet_list,
        "taskList": parse_task_list,
        "codeBlock": parse_code_block,
        "table": parse_table,
        "image": lambda node: {"type": "image", "_raw": node, "id": (node.get("attrs") or {}).get("id")},
    }
    if ntype in dispatch:
        return dispatch[ntype](node)
    # Unknown / unsupported node — preserve type so nothing is lost
    return {"type": ntype, "_raw": node}


def tiptap_to_synnoia(tiptap_data: dict) -> dict:
    """Convert a full TipTap document dict to Synnoia JSON."""
    top_content = tiptap_data.get("content") or []
    nodes: list[dict] = []
    for raw_node in top_content:
        parsed = parse_node(raw_node)
        if parsed is not None:
            nodes.append(parsed)
    return {"nodes": nodes}

