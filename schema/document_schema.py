from __future__ import annotations
import uuid
from typing import Literal, Optional, Union, List, Set
from pydantic import BaseModel, Field, model_validator

def generate_id() -> str:
    return uuid.uuid4().hex[:10]

class Formatting(BaseModel):
    """
    Inline text formatting options.
    Only include fields that are actually applied — all are optional.
    """
    bold:          Optional[bool] = None
    italic:        Optional[bool] = None
    underline:     Optional[bool] = None
    strikethrough: Optional[bool] = None
    superscript:   Optional[bool] = None
    subscript:     Optional[bool] = None
    fontSize:      Optional[str]  = None   
    fontFamily:    Optional[str]  = None   
    color:         Optional[str]  = None   
    highlight:     Optional[str]  = None   
    link:          Optional[str]  = None   

    model_config = {"extra": "forbid"}

class Segment(BaseModel):
    """
    A run of text with optional formatting.
    A paragraph or list item is made of one or more segments.
    """
    text:        str                   = Field(..., description="Raw text content")
    is_citation: Optional[bool]        = Field(None, description="True for citation refs like [2], [35]")
    formatting:  Optional[Formatting]  = None

    model_config = {"extra": "forbid"}

class HeadingNode(BaseModel):
    """H1 / H2 / H3 section heading."""
    type:       Literal["heading"]
    level:      Literal[1, 2, 3]       = Field(..., description="1=H1, 2=H2, 3=H3")
    text:       str                    = Field(..., description="Heading text")
    id:         str                    = Field(default_factory=generate_id)
    toc_id:     Optional[str]          = Field(None, description="Table-of-contents ID")
    lineHeight: Optional[str]          = None
    textAlign:  Optional[Literal["start", "center", "end", "justify"]] = None
    indent:     Optional[int]          = None
    margin:     Optional[Margin]       = None
    formatting: Optional[Formatting]   = None

    model_config = {"extra": "forbid"}

class Margin(BaseModel):
    top:    Optional[str] = None
    bottom: Optional[str] = None

    model_config = {"extra": "forbid"}

class ParagraphNode(BaseModel):
    """Body text paragraph — one or more segments."""
    type:       Literal["paragraph"]
    id:         str                         = Field(default_factory=generate_id)
    textAlign:  Optional[Literal["start", "center", "end", "justify"]] = None
    lineHeight: Optional[Union[float, str]]  = None
    indent:     Optional[int]                = None
    margin:     Optional[Margin]             = None
    segments:   Optional[List[Segment]]     = None

    model_config = {"extra": "forbid"}

class BlockquoteNode(BaseModel):
    """Blockquote / pull-quote block."""
    type:     Literal["blockquote"]
    id:       str                 = Field(default_factory=generate_id)
    segments: List[Segment]

    model_config = {"extra": "forbid"}

class ListItem(BaseModel):
    """One item inside an ordered or bullet list."""
    id:       str               = Field(default_factory=generate_id)
    index:    Optional[int]    = Field(None, description="Item number (ordered lists only)")
    segments: List[Segment]

    model_config = {"extra": "forbid"}

class OrderedListNode(BaseModel):
    """Numbered list."""
    type:     Literal["ordered_list"]
    id:       str               = Field(default_factory=generate_id)
    listType: Literal["decimal", "lower-alpha", "upper-alpha", "lower-roman"] = "decimal"
    items:    List[ListItem]

    model_config = {"extra": "forbid"}

class BulletListNode(BaseModel):
    """Unordered / bullet list."""
    type:     Literal["bullet_list"]
    id:       str               = Field(default_factory=generate_id)
    listType: Literal["disc", "circle", "square"] = "disc"
    items:    List[ListItem]

    model_config = {"extra": "forbid"}

class TaskItem(BaseModel):
    """Single checkbox item inside a task list."""
    id:      str               = Field(default_factory=generate_id)
    checked: bool
    text:    str

    model_config = {"extra": "forbid"}

class TaskListNode(BaseModel):
    """Checklist / task list."""
    type:  Literal["task_list"]
    id:    str               = Field(default_factory=generate_id)
    items: List[TaskItem]

    model_config = {"extra": "forbid"}

class CodeBlockNode(BaseModel):
    """Preformatted code block."""
    type:     Literal["code_block"]
    id:       str                                      = Field(default_factory=generate_id)
    code:     str                                       = Field(..., description="Code content")
    language: Optional[str]                             = "plaintext"
    theme:    Optional[Literal["light", "dark"]]        = "light"
    wordWrap: Optional[bool]                            = True

    model_config = {"extra": "forbid"}

class TableCell(BaseModel):
    """Single cell in a table row."""
    id:         str              = Field(default_factory=generate_id)
    text:       str
    colspan:    Optional[int]  = 1
    rowspan:    Optional[int]  = 1
    align:      Optional[Literal["left", "center", "right"]] = None
    background: Optional[str]  = None   # hex color
    color:      Optional[str]  = None   # hex color

    model_config = {"extra": "forbid"}

class TableRow(BaseModel):
    """One row in a table."""
    id:       str               = Field(default_factory=generate_id)
    row_type: Literal["header", "body"]
    cells:    List[TableCell]

    model_config = {"extra": "forbid"}

class TableNode(BaseModel):
    """Data table with header and body rows."""
    type: Literal["table"]
    id:   str               = Field(default_factory=generate_id)
    rows: List[TableRow]

    model_config = {"extra": "forbid"}

Node = Union[
    HeadingNode,
    ParagraphNode,
    BlockquoteNode,
    OrderedListNode,
    BulletListNode,
    TaskListNode,
    CodeBlockNode,
    TableNode,
]

class DocumentBody(BaseModel):
    nodes: List[Node] = Field(..., description="All document nodes in reading order")

    @model_validator(mode="after")
    def ensure_unique_ids(self) -> "DocumentBody":
        seen: Set[str] = set()

        def fix(node_id: str) -> str:
            if node_id in seen:
                return generate_id()  # duplicate → replace silently
            seen.add(node_id)
            return node_id

        for node in self.nodes:
            node.id = fix(node.id)
            if hasattr(node, "items"):
                for item in node.items:
                    item.id = fix(item.id)
            if hasattr(node, "rows"):
                for row in node.rows:
                    row.id = fix(row.id)
                    for cell in row.cells:
                        cell.id = fix(cell.id)

        return self

    model_config = {"extra": "forbid"}

class Document(BaseModel):
    """Root document model — use this as the LLM structured output type."""
    document: DocumentBody

    model_config = {"extra": "forbid"}

