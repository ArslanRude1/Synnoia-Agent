from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os

load_dotenv()

from pathlib import Path
import sys

project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from schema.xml_schema import MxGraphModel

class DiagramResponse(BaseModel):
    diagram_type: str = Field(description="Type of diagram e.g. flowchart, mindmap, timeline, orgchart, sequence, er_diagram, decision_tree, network, etc.")
    title: str = Field(description="Title of the diagram")
    graph: MxGraphModel = Field(description="Valid draw.io mxGraphModel XML")
    action_summary: str = Field(description="What diagram was created, what it represents, node count, layout direction")

model = ChatOpenAI(
    model="gpt-5.4",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7,
    max_retries=2,
    reasoning_effort="medium",
    streaming=True
)

model_with_structured_output = model.with_structured_output(DiagramResponse)

edit_prompt = ChatPromptTemplate([
    ("system", '''You are a Diagram Generation Agent. You produce valid draw.io XML from any diagram request — you are not limited to specific diagram types. You infer the most appropriate diagram type from the request and generate clean, well-structured draw.io XML.
    ─────────────────────────────────────────────────────────────────────────────
    DIAGRAM TYPE SELECTION
    ─────────────────────────────────────────────────────────────────────────────
    You are not limited to a fixed list. Choose the diagram type that best represents the content:
    - Process or steps with decisions → flowchart
    - Hierarchical topic breakdown → mindmap or concept_map
    - Chronological events → timeline
    - Organizational hierarchy → orgchart
    - System interactions over time → sequence
    - Database structure → er_diagram
    - Binary branching choices → decision_tree
    - Infrastructure or system topology → network
    - Parallel processes with actors → swimlane
    - Any other structure → name it appropriately and generate accordingly

    If the user names a specific diagram type, use exactly that.
    If the user describes content without naming a type, infer the most appropriate one.

    CRITICAL XML RULES:
    - id="0" and id="1" are ALWAYS reserved for root cells — never use them for content nodes
    - All content mxCell elements must have parent="1"
    - Vertex cells must have vertex="1"
    - Edge cells must have edge="1" plus source and target attributes referencing valid node ids
    - Every edge source and target must reference an existing node id
    - Node ids must be unique — use integers starting from 2, or short unique strings
    - All attribute values must be properly quoted
    - The xml value in your JSON output must be a valid escaped string

    ─────────────────────────────────────────────────────────────────────────────
    NODE STYLES — use these for common shapes
    ─────────────────────────────────────────────────────────────────────────────

    Rectangle (process, default):
    style="rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;"

    Diamond (decision):
    style="rhombus;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;"

    Ellipse (start/end terminal):
    style="ellipse;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;"

    Parallelogram (input/output):
    style="shape=parallelogram;perimeter=parallelogramPerimeter;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;"

    Rounded rectangle (milestone/event):
    style="rounded=1;arcSize=50;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;"

    Cylinder (database/storage):
    style="shape=mxgraph.flowchart.database;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;"

    Cloud (external system):
    style="shape=mxgraph.cisco.sites.generic_building;whiteSpace=wrap;html=1;"

    Edge (default arrow):
    style="edgeStyle=orthogonalEdgeStyle;html=1;"

    Edge (curved):
    style="edgeStyle=elbowEdgeStyle;elbow=vertical;html=1;"

    Edge (dotted):
    style="edgeStyle=orthogonalEdgeStyle;html=1;dashed=1;"

    ─────────────────────────────────────────────────────────────────────────────
    LAYOUT RULES
    ─────────────────────────────────────────────────────────────────────────────
    TB (top-bottom) — default for most diagrams:
    - Start y=40, increment y by 120 per row
    - Center nodes horizontally around x=400
    - Width=120, Height=60 for rectangles; Width=120, Height=80 for diamonds

    LR (left-right) — for timelines, sequence, horizontal flows:
    - Start x=40, increment x by 200 per column
    - Center nodes vertically around y=300

    Radial (mindmap, concept map):
    - Central node at x=400, y=300, Width=160, Height=60
    - First level branches: place at distance 220 in 8 compass directions
    - Second level: extend 160px further from their parent

    Spacing rules:
    - Minimum 80px gap between any two nodes
    - Diamonds need 40px extra vertical space above and below
    - Labels on edges: max 3 words
    - Node labels: max 6 words — truncate or abbreviate if needed

    ─────────────────────────────────────────────────────────────────────────────
    CONTENT RULES
    ─────────────────────────────────────────────────────────────────────────────
    - If a document is provided, extract actual content — use real section names, process steps, entities, concepts, and relationships from the document
    - If no document is provided, generate a complete and meaningful diagram purely from the request
    - Do not generate placeholder labels like "Node 1", "Step A" — use real descriptive labels
    - Do not generate empty or near-empty diagrams — minimum 4 nodes for any diagram
    - For complex documents, prioritize the most important relationships — do not try to map everything
    - If the outline contains validation, check, or decision steps, always use a diamond (rhombus) node with Yes/No edges branching out

    ─────────────────────────────────────────────────────────────────────────────
    ACTION SUMMARY RULES
    ─────────────────────────────────────────────────────────────────────────────
    - Always 5-9 sentences
    - Mention: diagram type chosen, what it represents, number of nodes, layout direction
    - Be specific — never write 'The diagram was created successfully'
    - Example: 'Generated a top-down flowchart with 9 nodes illustrating the 6-step loan approval process, from application submission through credit check to final disbursement.'''),
    ("user", '''Request: {instructions}
    Diagram plan from planner: {diagram_outline}
    Document (extract content if relevant): {doc_text}
    Generate draw.io XML:''')
])

diagram_chain = edit_prompt | model_with_structured_output

# if __name__ == "__main__":
#     initial_state = {
#     "instructions": """
# Create a detailed system architecture diagram for an e-commerce order processing system.
# Show all components, decision points, parallel processes, and failure paths.
# Use appropriate shapes for each component type.
# Include database nodes, external services, and decision diamonds.
# """,
#     "outline": {
#         "title": "E-Commerce Order Processing System",
#         "diagram_type": "flowchart",
#         "direction": "TB",
#         "components": {
#             "entry_points": [
#                 "Customer Places Order",
#                 "Mobile App",
#                 "Web Browser",
#                 "Third Party API"
#             ],
#             "processing_steps": [
#                 "API Gateway",
#                 "Authentication Service",
#                 "Order Validation",
#                 "Inventory Check",
#                 "Payment Processing",
#                 "Fraud Detection",
#                 "Order Confirmation",
#                 "Warehouse Management",
#                 "Shipping Service",
#                 "Notification Service"
#             ],
#             "decision_nodes": [
#                 "Auth Valid?",
#                 "Order Valid?",
#                 "In Stock?",
#                 "Payment Approved?",
#                 "Fraud Detected?"
#             ],
#             "databases": [
#                 "User Database",
#                 "Order Database",
#                 "Inventory Database",
#                 "Payment Records"
#             ],
#             "external_services": [
#                 "Payment Gateway",
#                 "Email Service",
#                 "SMS Service",
#                 "Shipping Provider"
#             ],
#             "failure_paths": [
#                 "Auth Failed → Return 401",
#                 "Validation Failed → Return 400",
#                 "Out of Stock → Notify Customer",
#                 "Payment Failed → Retry or Cancel",
#                 "Fraud Alert → Manual Review"
#             ],
#             "terminal_nodes": [
#                 "Order Complete",
#                 "Order Failed",
#                 "Order Pending Review"
#             ]
#         },
#         "connections": [
#             "Mobile App → API Gateway",
#             "Web Browser → API Gateway",
#             "Third Party API → API Gateway",
#             "API Gateway → Authentication Service",
#             "Authentication Service → Auth Valid?",
#             "Auth Valid? Yes → Order Validation",
#             "Auth Valid? No → Return 401",
#             "Order Validation → Order Valid?",
#             "Order Valid? Yes → Inventory Check",
#             "Order Valid? No → Return 400",
#             "Inventory Check → Inventory Database",
#             "Inventory Check → In Stock?",
#             "In Stock? Yes → Fraud Detection",
#             "In Stock? No → Notify Customer",
#             "Fraud Detection → Fraud Detected?",
#             "Fraud Detected? Yes → Manual Review",
#             "Fraud Detected? No → Payment Processing",
#             "Payment Processing → Payment Gateway",
#             "Payment Processing → Payment Approved?",
#             "Payment Approved? Yes → Order Confirmation",
#             "Payment Approved? No → Payment Failed",
#             "Order Confirmation → Order Database",
#             "Order Confirmation → Warehouse Management",
#             "Warehouse Management → Shipping Service",
#             "Shipping Service → Shipping Provider",
#             "Order Confirmation → Notification Service",
#             "Notification Service → Email Service",
#             "Notification Service → SMS Service",
#             "Notification Service → Order Complete"
#         ]
#     },
#     "doc_text": ""
# }

#     result = diagram_chain.invoke(initial_state)

#     print(f"Type:    {result.diagram_type}")
#     print(f"Title:   {result.title}")
#     print(f"Summary: {result.action_summary}")
#     print(f"Nodes:   {len([c for c in result.graph.root.cells if c.vertex == '1'])}")
#     print()
#     print(result.graph.to_xml())