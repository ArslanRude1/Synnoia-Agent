from __future__ import annotations

import json
import sys
from typing import Any


def load_synnoia(filepath: str) -> dict:
    """Load raw Synnoia JSON from *filepath*."""
    with open(filepath, encoding="utf-8") as fh:
        return json.load(fh)


def _formatting_to_marks(fmt: dict) -> list[dict]:
    """
    Convert a simple formatting dict back into TipTap mark objects.
    Preserves the original mark structure precisely.
    """
    marks: list[dict] = []

    # textStyle mark (fontFamily / fontSize / color)
    ts_attrs: dict[str, Any] = {
        "fontFamily": fmt.get("fontFamily", None),
        "fontSize": fmt.get("fontSize", None),
        "color": fmt.get("color", None),
    }
    if any(v is not None for v in ts_attrs.values()):
        marks.append({"type": "textStyle", "attrs": ts_attrs})

    # link mark — omitted (not stored in Simple JSON)

    # Simple boolean marks
    if fmt.get("bold"):
        marks.append({"type": "bold"})
    if fmt.get("italic"):
        marks.append({"type": "italic"})
    if fmt.get("strike") or fmt.get("strikethrough"):
        marks.append({"type": "strike"})
    if fmt.get("underline"):
        marks.append({"type": "underline"})
    if fmt.get("superscript"):
        marks.append({"type": "superscript"})
    if fmt.get("subscript"):
        marks.append({"type": "subscript"})

    # highlight
    highlight = fmt.get("highlight")
    if highlight:
        marks.append({"type": "highlight", "attrs": {"color": highlight}})

    return marks


def segments_to_tiptap_content(segments: list[dict]) -> list[dict]:
    """Convert simple ``segments`` list back to TipTap text nodes with marks."""
    nodes: list[dict] = []
    for seg in segments:
        text = seg.get("text") or ""
        fmt = seg.get("formatting") or {}
        marks = _formatting_to_marks(fmt)
        node: dict[str, Any] = {"type": "text", "text": text}
        if marks:
            node["marks"] = marks
        nodes.append(node)
    return nodes


def _default_para_attrs(**overrides) -> dict:
    base = {"indent": None, "textAlign": None, "lineHeight": 1.5, "margin": {}, "id": None}
    base.update(overrides)
    return base


def _default_cell_attrs(extra: dict | None = None) -> dict:
    base: dict[str, Any] = {
        "colspan": 1,
        "rowspan": 1,
        "colwidth": None,
        "align": None,
        "background": None,
        "color": None,
    }
    if extra:
        base.update(extra)
    return base


def _wrap_in_paragraph(content_nodes: list[dict], para_attrs: dict | None = None) -> dict:
    attrs = para_attrs or _default_para_attrs()
    node: dict[str, Any] = {"type": "paragraph", "attrs": attrs}
    if content_nodes:
        node["content"] = content_nodes
    return node


def synnoia_to_tiptap_node(node: dict) -> dict | None:  # noqa: C901
    """Convert a single synnoia-format node back to TipTap format."""
    ntype = node.get("type") or ""

    # ── Heading ─────────────────────────────────────────────────────
    if ntype == "heading":
        level = node.get("level") or 2
        node_id = node.get("id")
        toc_id = node.get("toc_id")
        fmt = node.get("formatting") or {}
        text = node.get("text") or ""

        # Reconstruct attrs
        attrs: dict[str, Any] = {
            "indent": node.get("indent"),
            "textAlign": node.get("textAlign"),
            "lineHeight": node.get("lineHeight") or "1.375",
            "margin": node.get("margin") or {},
            "id": node_id,
            "data-toc-id": toc_id,
            "level": level,
        }

        # Build text node
        marks = _formatting_to_marks(fmt)
        text_node: dict[str, Any] = {"type": "text", "text": text}
        if marks:
            text_node["marks"] = marks

        return {"type": "heading", "attrs": attrs, "content": [text_node]}

    # ── Paragraph ───────────────────────────────────────────────────
    if ntype == "paragraph":
        attrs = {
            "indent": node.get("indent"),
            "textAlign": node.get("textAlign"),
            "lineHeight": node.get("lineHeight") or 1.5,
            "margin": node.get("margin") or {},
            "id": node.get("id"),
        }
        segments = node.get("segments") or []
        content_nodes = segments_to_tiptap_content(segments)
        result: dict[str, Any] = {"type": "paragraph", "attrs": attrs}
        if content_nodes:
            result["content"] = content_nodes
        return result

    # ── Blockquote ──────────────────────────────────────────────────
    if ntype == "blockquote":
        segments = node.get("segments") or []
        para_attrs = _default_para_attrs(textAlign="start")
        inner_para = _wrap_in_paragraph(segments_to_tiptap_content(segments), para_attrs)
        result: dict[str, Any] = {"type": "blockquote", "content": [inner_para]}
        node_id = node.get("id")
        if node_id:
            result["attrs"] = {"id": node_id}
        return result

    # ── Ordered list ────────────────────────────────────────────────
    if ntype == "ordered_list":
        list_type = node.get("listType") or "decimal"
        items_raw = node.get("items") or []
        list_items: list[dict] = []
        for item in items_raw:
            segs = item.get("segments") or []
            item_id = item.get("id")
            # Each listItem in TipTap wraps paragraph(s)
            # We keep a single paragraph per item (restoring split paragraphs
            # exactly is impossible without original split markers, so one para)
            para_attrs = _default_para_attrs(textAlign="start")
            inner = _wrap_in_paragraph(segments_to_tiptap_content(segs), para_attrs)
            list_item_attrs: dict[str, Any] = {"indent": None}
            if item_id:
                list_item_attrs["id"] = item_id
            list_items.append({
                "type": "listItem",
                "attrs": list_item_attrs,
                "content": [inner],
            })
        result: dict[str, Any] = {
            "type": "orderedList",
            "attrs": {"margin": {}, "start": 1, "listType": list_type},
            "content": list_items,
        }
        node_id = node.get("id")
        if node_id:
            result["attrs"]["id"] = node_id
        return result

    # ── Bullet list ─────────────────────────────────────────────────
    if ntype == "bullet_list":
        list_type = node.get("listType") or "disc"
        items_raw = node.get("items") or []
        list_items = []
        for item in items_raw:
            segs = item.get("segments") or []
            item_id = item.get("id")
            para_attrs = _default_para_attrs()
            inner = _wrap_in_paragraph(segments_to_tiptap_content(segs), para_attrs)
            list_item_attrs: dict[str, Any] = {"indent": None}
            if item_id:
                list_item_attrs["id"] = item_id
            list_items.append({
                "type": "listItem",
                "attrs": list_item_attrs,
                "content": [inner],
            })
        result: dict[str, Any] = {
            "type": "bulletList",
            "attrs": {"margin": {}, "listType": list_type},
            "content": list_items,
        }
        node_id = node.get("id")
        if node_id:
            result["attrs"]["id"] = node_id
        return result

    # ── Task list ───────────────────────────────────────────────────
    if ntype == "task_list":
        items_raw = node.get("items") or []
        task_items: list[dict] = []
        for item in items_raw:
            checked = item.get("checked") or False
            text_val = item.get("text") or ""
            item_id = item.get("id")
            inner_para = _wrap_in_paragraph(
                [{"type": "text", "text": text_val}],
                _default_para_attrs(textAlign="start"),
            )
            task_item_attrs: dict[str, Any] = {"indent": None, "checked": checked}
            if item_id:
                task_item_attrs["id"] = item_id
            task_items.append({
                "type": "taskItem",
                "attrs": task_item_attrs,
                "content": [inner_para],
            })
        result: dict[str, Any] = {
            "type": "taskList",
            "attrs": {"margin": {}},
            "content": task_items,
        }
        node_id = node.get("id")
        if node_id:
            result["attrs"]["id"] = node_id
        return result

    # ── Code block ──────────────────────────────────────────────────
    if ntype == "code_block":
        attrs = {
            "margin": {},
            "language": node.get("language") or "plaintext",
        }
        node_id = node.get("id")
        if node_id:
            attrs["id"] = node_id
        theme = node.get("theme")
        if theme:
            attrs["theme"] = theme
        word_wrap = node.get("wordWrap")
        if word_wrap is not None:
            attrs["wordWrap"] = word_wrap
        code = node.get("code") or ""
        return {
            "type": "codeBlock",
            "attrs": attrs,
            "content": [{"type": "text", "text": code}],
        }

    # ── Table ───────────────────────────────────────────────────────
    if ntype == "table":
        rows_simple = node.get("rows") or []
        tiptap_rows: list[dict] = []

        for row in rows_simple:
            row_type = row.get("row_type") or "body"
            row_id = row.get("id")
            cell_tag = "tableHeader" if row_type == "header" else "tableCell"
            cells_simple = row.get("cells") or []
            tiptap_cells: list[dict] = []

            for cell_obj in cells_simple:
                if isinstance(cell_obj, str):
                    # Simplified format: cell is just a string
                    cell_text = cell_obj
                    segments: list[dict] = []
                    extra_attrs: dict = {}
                else:
                    cell_text = cell_obj.get("text") or ""
                    segments = cell_obj.get("segments") or []
                    extra_attrs = {
                        k: cell_obj.get(k)
                        for k in ("colspan", "rowspan", "align", "background",
                                  "color", "colwidth")
                        if k in cell_obj and cell_obj.get(k) is not None
                    }
                    # Extract cell id
                    cell_id = cell_obj.get("id")
                    if cell_id:
                        extra_attrs["id"] = cell_id

                cell_attrs = _default_cell_attrs(extra_attrs)

                if segments:
                    inner_content = segments_to_tiptap_content(segments)
                else:
                    inner_content = [{"type": "text", "text": cell_text}]

                para_attrs = _default_para_attrs()
                inner_para = _wrap_in_paragraph(inner_content, para_attrs)

                tiptap_cells.append({
                    "type": cell_tag,
                    "attrs": cell_attrs,
                    "content": [inner_para],
                })

            row_obj: dict[str, Any] = {"type": "tableRow", "content": tiptap_cells}
            if row_id:
                row_obj["attrs"] = {"id": row_id}
            tiptap_rows.append(row_obj)

        result: dict[str, Any] = {
            "type": "table",
            "attrs": {"margin": {}},
            "content": tiptap_rows,
        }
        node_id = node.get("id")
        if node_id:
            result["attrs"]["id"] = node_id
        return result

    # ── Image ─────────────────────────────────────────────────────────
    if ntype == "image":
        raw = node.get("_raw")
        if raw:
            return raw
        return None

    # ── Unknown / raw passthrough ────────────────────────────────────
    raw = node.get("_raw")
    if raw:
        return raw
    return None


def synnoia_to_tiptap(synnoia_data: dict) -> dict:
    """Convert a full Synnoia JSON document back to TipTap format."""
    # Handle both {"document": {"nodes": [...]}} and {"nodes": [...]} formats
    if "document" in synnoia_data:
        nodes_simple = synnoia_data.get("document", {}).get("nodes") or []
    else:
        nodes_simple = synnoia_data.get("nodes") or []
    
    content: list[dict] = []
    for snode in nodes_simple:
        tt = synnoia_to_tiptap_node(snode)
        if tt is not None:
            content.append(tt)
    return {"type": "doc", "content": content}

