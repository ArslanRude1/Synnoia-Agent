import sys
import warnings
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from schema.document_schema import Document
from schema.xml_schema import MxGraphModel

class SynnoiaState(BaseModel):
    query: str = Field(default="")
    doc_text: str = Field(default="")
    doc_json: str = Field(default="")
    model: str = Field(default="gpt-5.4")

    response: Optional[str] = Field(default="")

    intent: str = Field(default="")
    rephrased_query: str = Field(default="")
    clarification_question: Optional[str] = Field(None)
    chitchat_response: Optional[str] = Field(None)

    operation_type: str = Field(default="")
    anchor_id: Optional[str] = Field(default=None)
    response_json: Optional[Document] = Field(default=None)
    diagram_type: str = Field(default="")
    title: str = Field(default="")
    graph: Optional[MxGraphModel] = Field(default=None)
    action_summary: str = Field(default="")

synnoia_graph = StateGraph(SynnoiaState)

async def router_agent(state: SynnoiaState):
    from agents.router_agent.agent import router_chain
    result = await router_chain.ainvoke({"query": state.query,"doc_text":state.doc_text})
    return {
        "rephrased_query": result.rephrased_query,
        "intent": result.intent,
        "clarification_question": result.clarification_question,
        "chitchat_response": result.chitchat_response
    }

async def communication_agent(state: SynnoiaState):
    from agents.communication_agent.agent import communication_chain
    result = await communication_chain.ainvoke({"rephrased_query": state.rephrased_query, "doc_text": state.doc_text})
    return {
        "response": result.content if hasattr(result, "content") else result,
    }  

from generation_graph.graph import generation_agent

def router(state: SynnoiaState):
    intent = state.intent
    if intent == "communication":
        return "communication_agent"
    elif intent == "generation":
        return "generation_graph"
    elif intent == "chitchat":
        return "chitchat"
    elif intent == "clarification":
        return "clarification"

synnoia_graph.add_node("router_agent", router_agent)
synnoia_graph.add_node("communication_agent", communication_agent)
synnoia_graph.add_node("generation_graph", generation_agent)


synnoia_graph.add_edge(START, "router_agent")
synnoia_graph.add_conditional_edges("router_agent", router,{
    "communication_agent": "communication_agent",
    "generation_graph": "generation_graph",
    "chitchat": END,
    "clarification": END
})
synnoia_graph.add_edge("communication_agent", END)
synnoia_graph.add_edge("generation_graph", END)

synnoia_agent = synnoia_graph.compile()


# if __name__ == "__main__":
#     initial_state = {
#         "query": "make a flowchart on ai revolution",
#         "doc_text": "",
#         "doc_json": "",
#         "model": "gpt-5.4",
#         "intent": "",
#         "rephrased_query": "",
#         "chitchat_response": "",
#         "clarification_question": "",
#         "operation_type": "",
#         "anchor_id": None,
#         "response_json": None,
#         "diagram_type": "",
#         "title": "",
#         "graph": None,
#         "action_summary": "",
#     }
    
#     for chunk in synnoia_agent.stream(initial_state, stream_mode="messages", version="v2", subgraphs=True):
#         if chunk["type"] == "messages":
#             message_chunk, metadata = chunk["data"]
#             if message_chunk.content:
#                 print(message_chunk.content, end="", flush=True)
#     print("\n")

