from pydantic import BaseModel, Field
from typing import List, Optional, Literal


# ── Write Outline ──────────────────────────────────────────────────────────────
class WriteOutlineItem(BaseModel):
    section:      str       = Field(description="Section title")
    instructions: str       = Field(description="What this section should contain — 1-2 sentences")
    node_types:   List[str] = Field(default_factory=list, description="e.g. heading, paragraph, bullet_list")

    model_config = {"extra": "forbid"}


# ── Diagram Outline ────────────────────────────────────────────────────────────
class DiagramNode(BaseModel):
    id:    str = Field(description="Unique node id starting from 2")
    label: str = Field(description="Display label — max 6 words")
    shape: Literal[
        "rectangle", "diamond", "ellipse",
        "parallelogram", "cylinder", "cloud", "rounded_rectangle"
    ]

    model_config = {"extra": "forbid"}


class DiagramEdge(BaseModel):
    from_id: str           = Field(description="Source node id")
    to_id:   str           = Field(description="Target node id")
    label:   Optional[str] = Field(default=None, description="Edge label — max 3 words")

    model_config = {"extra": "forbid"}


class DiagramOutlineItem(BaseModel):
    diagram_type: str       = Field(description="flowchart, mindmap, timeline, orgchart, sequence, er_diagram, swimlane, network etc.")
    direction:    Literal["TB", "LR", "RL", "BT"] = Field(default="TB")
    nodes:        List[DiagramNode] = Field(description="All nodes in the diagram")
    edges:        List[DiagramEdge] = Field(description="All connections between nodes")

    model_config = {"extra": "forbid"}