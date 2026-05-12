from pydantic import BaseModel, Field, model_validator
from typing import Optional, List


# ── Geometry ───────────────────────────────────────────────────────────────────
class MxPoint(BaseModel):
    x: Optional[float] = Field(default=None)
    y: Optional[float] = Field(default=None)


class MxGeometry(BaseModel):
    x:        Optional[float]      = Field(default=None)
    y:        Optional[float]      = Field(default=None)
    width:    Optional[float]      = Field(default=None)
    height:   Optional[float]      = Field(default=None)
    relative: Optional[str]        = Field(default=None, description="'1' for edge geometries")
    as_:      Optional[str]        = Field(default="geometry", alias="as")
    points:   Optional[List[MxPoint]] = Field(default=None, description="Waypoints for curved edges")

    model_config = {"populate_by_name": True}


# ── Cell ───────────────────────────────────────────────────────────────────────
class MxCell(BaseModel):
    id:       str                    = Field(description="Unique cell id — '0' and '1' reserved for root")
    value:    str                    = Field(default="", description="Display label — empty string for edges with no label")
    style:    Optional[str]          = Field(default=None, description="draw.io style string")
    vertex:   Optional[str]          = Field(default=None, description="'1' for shape nodes")
    edge:     Optional[str]          = Field(default=None, description="'1' for connections")
    parent:   Optional[str]          = Field(default="1", description="Always '1' for content cells")
    source:   Optional[str]          = Field(default=None, description="Source node id — required for edges")
    target:   Optional[str]          = Field(default=None, description="Target node id — required for edges")
    geometry: Optional[MxGeometry]   = Field(default=None, description="Position and size — required for vertex cells")


# ── Root ───────────────────────────────────────────────────────────────────────
class MxRoot(BaseModel):
    cells: List[MxCell] = Field(description="All mxCell elements including reserved root cells")

    @model_validator(mode="after")
    def validate_cells(self) -> "MxRoot":
        cells  = self.cells
        ids    = [c.id for c in cells]
        id_set = set(ids)

        # 1 — reserved root cells must exist
        if "0" not in id_set:
            raise ValueError("Missing reserved cell id='0'")
        if "1" not in id_set:
            raise ValueError("Missing reserved cell id='1'")

        # 2 — no duplicate ids
        seen = set()
        for id_ in ids:
            if id_ in seen:
                raise ValueError(f"Duplicate mxCell id='{id_}'")
            seen.add(id_)

        # 3 — edge validation
        for cell in cells:
            if cell.edge == "1":
                if not cell.source:
                    raise ValueError(f"Edge id='{cell.id}' missing source")
                if not cell.target:
                    raise ValueError(f"Edge id='{cell.id}' missing target")
                if cell.source not in id_set:
                    raise ValueError(f"Edge id='{cell.id}' source='{cell.source}' not found")
                if cell.target not in id_set:
                    raise ValueError(f"Edge id='{cell.id}' target='{cell.target}' not found")

        # 4 — vertex cells must have geometry
        for cell in cells:
            if cell.vertex == "1" and cell.id not in ("0", "1"):
                if cell.geometry is None:
                    raise ValueError(f"Vertex id='{cell.id}' missing geometry")
                if cell.geometry.x is None or cell.geometry.y is None:
                    raise ValueError(f"Vertex id='{cell.id}' geometry missing x or y")
                if cell.geometry.width is None or cell.geometry.height is None:
                    raise ValueError(f"Vertex id='{cell.id}' geometry missing width or height")

        # 5 — minimum content nodes
        content = [c for c in cells if c.id not in ("0", "1") and c.vertex == "1"]
        if len(content) < 4:
            raise ValueError(f"Only {len(content)} content nodes — minimum is 4")

        return self


# ── Graph Model ────────────────────────────────────────────────────────────────
class MxGraphModel(BaseModel):
    dx:         str = Field(default="1422")
    dy:         str = Field(default="762")
    grid:       str = Field(default="1")
    gridSize:   str = Field(default="10")
    guides:     str = Field(default="1")
    tooltips:   str = Field(default="1")
    connect:    str = Field(default="1")
    arrows:     str = Field(default="1")
    fold:       str = Field(default="1")
    page:       str = Field(default="1")
    pageScale:  str = Field(default="1")
    pageWidth:  str = Field(default="1169")
    pageHeight: str = Field(default="827")
    math:       str = Field(default="0")
    shadow:     str = Field(default="0")
    root:       MxRoot = Field(description="Contains all mxCell elements")

    def to_xml(self) -> str:
        """Convert the Pydantic model back to draw.io XML string."""
        lines = []
        lines.append(
            f'<mxGraphModel dx="{self.dx}" dy="{self.dy}" grid="{self.grid}" '
            f'gridSize="{self.gridSize}" guides="{self.guides}" tooltips="{self.tooltips}" '
            f'connect="{self.connect}" arrows="{self.arrows}" fold="{self.fold}" '
            f'page="{self.page}" pageScale="{self.pageScale}" pageWidth="{self.pageWidth}" '
            f'pageHeight="{self.pageHeight}" math="{self.math}" shadow="{self.shadow}">'
        )
        lines.append("  <root>")

        for cell in self.root.cells:
            attrs = f'id="{cell.id}" value="{cell.value}"'
            if cell.style:   attrs += f' style="{cell.style}"'
            if cell.vertex:  attrs += f' vertex="{cell.vertex}"'
            if cell.edge:    attrs += f' edge="{cell.edge}"'
            if cell.parent:  attrs += f' parent="{cell.parent}"'
            if cell.source:  attrs += f' source="{cell.source}"'
            if cell.target:  attrs += f' target="{cell.target}"'

            if cell.geometry is None:
                lines.append(f'    <mxCell {attrs} />')
            else:
                lines.append(f'    <mxCell {attrs}>')
                geo = cell.geometry
                geo_attrs = f'as="geometry"'
                if geo.x is not None: geo_attrs += f' x="{int(geo.x)}"'
                if geo.y is not None: geo_attrs += f' y="{int(geo.y)}"'
                if geo.width is not None:  geo_attrs += f' width="{geo.width}"'
                if geo.height is not None: geo_attrs += f' height="{geo.height}"'
                if geo.relative:           geo_attrs += f' relative="{geo.relative}"'

                if geo.points:
                    lines.append(f'      <mxGeometry {geo_attrs}>')
                    lines.append('        <Array as="points">')
                    for pt in geo.points:
                        lines.append(f'          <mxPoint x="{pt.x}" y="{pt.y}" />')
                    lines.append('        </Array>')
                    lines.append('      </mxGeometry>')
                else:
                    lines.append(f'      <mxGeometry {geo_attrs} />')

                lines.append('    </mxCell>')

        lines.append("  </root>")
        lines.append("</mxGraphModel>")
        return "\n".join(lines)

