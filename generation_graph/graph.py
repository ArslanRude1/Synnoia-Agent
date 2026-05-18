import sys
import warnings
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from schema.document_schema import Document
from schema.outline_schema import WriteOutlineItem, DiagramOutlineItem
from schema.xml_schema import MxGraphModel

class GenerationState(BaseModel):
    rephrased_query: str = Field(default="")
    doc_text: str = Field(default="")
    doc_json: str = Field(default="")
    model: str = Field(default="gpt-5.4")
    
    task_type: str = Field(default="")
    write_outline: Optional[List[WriteOutlineItem]] = Field(default_factory=list)
    diagram_outline: Optional[List[DiagramOutlineItem]] = Field(default_factory=list)
    exclude: Optional[List[str]] = Field(default_factory=list)

    operation_type: str = Field(default="")
    anchor_id: Optional[str] = Field(default=None)
    response_json: Optional[Document] = Field(default=None)
    action_summary: str = Field(default="")

    diagram_type: Optional[str] = Field(default="")
    title: Optional[str] = Field(default="")
    graph: Optional[MxGraphModel] = Field(default=None)

generation_graph = StateGraph(GenerationState)

async def planner_agent(state: GenerationState):
    from generation_graph.agents.planner_agent.agent import planner_chain
    result = await planner_chain.ainvoke({
        "rephrased_query": state.rephrased_query,
        "doc_json": state.doc_json
    })
    return {
        "task_type": result.task_type,
        "operation_type": result.operation_type,
        "anchor_id": result.anchor_id,
        "write_outline": result.write_outline,
        "diagram_outline": result.diagram_outline,
        "exclude": result.exclude
    }
def route(state: GenerationState):
    if state.task_type == "write":
        return "writer"
    elif state.task_type == "edit":
        return "edit"
    elif state.task_type == "humanize":
        return "humanizer"
    elif state.task_type == "deplagiarize":
        return "deplagiarizer"
    elif state.task_type == "diagram":
        return "diagram"
async def writer_agent(state: GenerationState):
    from generation_graph.agents.writer_agent.agent import writer_chain
    result = await writer_chain.ainvoke({
        "instructions": state.rephrased_query,
        "write_outline": state.write_outline,
        "doc_json": state.doc_json,
        "exclude": state.exclude
    })
    return {
        "response_json": result.document,
        "action_summary": result.action_summary
    }
async def edit_agent(state: GenerationState):
    from generation_graph.agents.edit_agent.agent import edit_chain
    result = await edit_chain.ainvoke({
        "instructions": state.rephrased_query,
        "doc_json": state.doc_json
    })
    return {
        "response_json": result.document,
        "action_summary": result.action_summary
    }
async def humanizer_agent(state: GenerationState):
    from generation_graph.agents.humanizer_agent.agent import humanizer_chain
    result = await humanizer_chain.ainvoke({
        "instructions": state.rephrased_query,
        "doc_json": state.doc_json
    })
    return {
        "response_json": result.document,
        "action_summary": result.action_summary
    }
async def deplagiarizer_agent(state: GenerationState):
    from generation_graph.agents.deplagiarizer_agent.agent import deplagiarizer_chain
    result = await deplagiarizer_chain.ainvoke({
        "instructions": state.rephrased_query,
        "doc_json": state.doc_json
    })
    return {
        "response_json": result.document,
        "action_summary": result.action_summary
    }
async def diagram_agent(state: GenerationState):
    from generation_graph.agents.diagram_agent.agent import diagram_chain
    result = await diagram_chain.ainvoke({
        "instructions": state.rephrased_query,
        "diagram_outline": state.diagram_outline,
        "doc_text": state.doc_text
    })
    return {
        "diagram_type": result.diagram_type,
        "title": result.title,
        "graph": result.graph,
        "action_summary": result.action_summary
    }

generation_graph.add_node("planner", planner_agent)
generation_graph.add_node("writer", writer_agent)
generation_graph.add_node("edit", edit_agent)
generation_graph.add_node("humanizer", humanizer_agent)
generation_graph.add_node("deplagiarizer", deplagiarizer_agent)
generation_graph.add_node("diagram", diagram_agent)

generation_graph.add_edge(START, "planner")
generation_graph.add_conditional_edges("planner", route,{
    "writer": "writer",
    "edit": "edit",
    "humanizer": "humanizer",
    "deplagiarizer": "deplagiarizer",
    "diagram": "diagram"
})
generation_graph.add_edge("writer", END)
generation_graph.add_edge("edit", END)
generation_graph.add_edge("humanizer", END)
generation_graph.add_edge("deplagiarizer", END)
generation_graph.add_edge("diagram", END)

generation_agent = generation_graph.compile()

