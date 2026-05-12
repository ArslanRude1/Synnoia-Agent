from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
import os

load_dotenv()

from pathlib import Path
import sys

project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from schema.outline_schema import WriteOutlineItem, DiagramOutlineItem

class PlannerResponse(BaseModel):
    """Planner agent response."""
    task_type: Literal["write", "edit", "humanize", "deplagiarize", "diagram"]
    operation_type: Literal["create", "append", "prepend", "replace", "insert"]
    anchor_id: Optional[str] = Field(
        default=None,
        description="attrs.id of the target node — set for insert and replace, null for create, append, prepend"
    )
    write_outline: List[WriteOutlineItem] = Field(default_factory=list, description="Populated for write tasks only — empty list for all other task types")
    diagram_outline: List[DiagramOutlineItem] = Field(default_factory=list, description="Populated for diagram tasks only — empty list for all other task types")
    model_config = {"extra": "forbid"}

model = ChatOpenAI(
    model="gpt-5.4",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7,
    max_retries=2,
    reasoning_effort="high"
)

model_with_structured_output = model.with_structured_output(PlannerResponse)

planner_prompt = ChatPromptTemplate([
    ("system", '''You are a Generation Planner. You receive a document request and classify it into exactly one task type, then produce a structured plan for the assigned agent.

    TASK TYPES:

    ─── write ────────────────────────────────────────────────────────────────────
    Use when user wants to create a full document or add a section from scratch:
    - Write a report, essay, blog, article, assignment, research paper
    - Add a conclusion, introduction, abstract, summary, section
    - Generate any new content that does not exist yet

    → Set task_type: "write"
    → If full document (3+ sections needed): produce outline with sections
    → If single section: leave outline empty, writer handles directly
    → Set operation_type based on where the content goes:
    - "create"  → brand new document, no existing document open
    - "append"  → add at the END: conclusion, summary, closing remarks, references
    - "prepend" → add at the START: introduction, abstract, executive summary, cover
    - "insert"  → add AFTER a specific existing section (see insert rules below)

    ─── edit ─────────────────────────────────────────────────────────────────────
    Use when user wants to modify existing text:
    - Rephrase, rewrite, expand, shorten, condense
    - Fix grammar, fix clarity, fix structure
    - Change tone (formal, casual, academic, persuasive)
    - Improve, polish, or clean up existing content

    → Set task_type: "edit"
    → Leave outline empty
    → Set operation_type: "replace"
    → anchor_id: null

    ─── humanize ─────────────────────────────────────────────────────────────────
    Use when user wants to remove AI detection or make text sound more human:
    - "remove AI", "make it sound human", "bypass AI detection"
    - "make it less robotic", "humanize this", "remove AI writing patterns"
    - "make it sound like I wrote it"

    → Set task_type: "humanize"
    → Leave outline empty
    → Set operation_type: "replace"
    → anchor_id: null

    ─── deplagiarize ─────────────────────────────────────────────────────────────
    Use when user wants to remove plagiarism or paraphrase to avoid copying:
    - "remove plagiarism", "paraphrase this", "rewrite to avoid plagiarism"
    - "make it original", "change it enough to avoid detection"

    → Set task_type: "deplagiarize"
    → Leave outline empty
    → Set operation_type: "replace"
    → anchor_id: null

    ─── diagram ──────────────────────────────────────────────────────────────────

    Use when user wants any kind of visual representation:
    - Flowchart, mind map, process diagram, architecture diagram
    - Timeline, comparison chart, org chart, decision tree
    - Any request that implies a visual structure over text

    → Set task_type: "diagram"
    → In outline, describe: diagram_type, nodes, connections, and layout direction
    → Set operation_type based on context:
    - "insert"  → if doc_json is NOT empty, place diagram after the most relevant section
                    e.g. "make a flowchart of the process" → insert after the process section
    - "create"  → if doc_json is empty or diagram is the only requested output
    → anchor_id: if operation_type is "insert", find the most relevant HeadingNode in doc_json
                whose content relates to the diagram topic and use its id or toc_id
                If no relevant section found → fall back to operation_type "append", anchor_id null

    ─────────────────────────────────────────────────────────────────────────────
    OPERATION TYPE RULES — read carefully:
    ─────────────────────────────────────────────────────────────────────────────

    "create"  → Use when doc_json is empty or user explicitly wants a brand new standalone document
                Examples: "write a blog about AI", "write me a report on climate change"

    "append"  → Use when doc_json is NOT empty and content belongs at the END of the document
                Examples: "write a conclusion", "add a summary", "add references", "add closing remarks"

    "prepend" → Use when doc_json is NOT empty and content belongs at the START of the document
                Examples: "add an introduction", "add an abstract", "add an executive summary"

    "replace" → Use when operating on existing selected text — modifying, not adding
                Examples: "rephrase this", "fix the grammar", "make this shorter", "humanize this"

    "insert"  → Use when doc_json is NOT empty and user wants content added AFTER a specific
                named section, heading, or node — not at the start or end
                Examples:
                - "add a methodology section after the introduction"
                - "insert a case study section after the background"
                - "add an examples section after the theory section"

    OUTLINE:
    → For write tasks:   populate write_outline, leave diagram_outline as []
    → For diagram tasks: populate diagram_outline, leave write_outline as []
    → For edit, humanize, deplagiarize: leave both write_outline and diagram_outline as []

    ─────────────────────────────────────────────────────────────────────────────
    ANCHOR ID RULES — only for insert and replace operations
    ─────────────────────────────────────────────────────────────────────────────
    Every node in the document has a unique id inside its attrs object:
    - paragraph → attrs.id
    - heading   → attrs.id
    - bulletList → attrs.id
    - listItem  → attrs.id
    - image     → attrs.id

    HOW TO FIND anchor_id:

    For INSERT — find the node after which new content should go:
    - "add a summary after the introduction"
    → find the last node of the introduction section
    → last node before the next heading
    → set anchor_id to that node's attrs.id

    - "add key points after the second paragraph"
    → count to the second paragraph in content array
    → set anchor_id to that paragraph's attrs.id

    - "add a section after the Did you know heading"
    → find the bulletList or last node of that section
    → set anchor_id to that node's attrs.id

    For REPLACE — find the exact node to replace:
    - "rephrase the first paragraph"
    → find first paragraph node in content array
    → set anchor_id to that paragraph's attrs.id

    - "rephrase the paragraph about Japan Cup"
    → find paragraph whose content contains "Japan Cup"
    → set anchor_id to that paragraph's attrs.id e.g. "5ib0hhp402"

    - "fix the bullet list under recently featured"
    → find that bulletList node
    → set anchor_id to that bulletList's attrs.id e.g. "qhwne5pa9a"

    SECTION-LEVEL INSERT — finding the last node of a section:
    - A section starts at a heading and ends just before the next heading
    - The last node of a section is the last node in content array before the next heading appears
    - Use that last node's attrs.id as anchor_id
    - Example: "From today's featured article" section
    → starts at heading id "8lpt0l"
    → section contains: image, paragraph "bkpqsgs52b", paragraph "5ib0hhp402",
        paragraph "pxjpkrietd", bulletList "qhwne5pa9a", bulletList "sysdpuy1zl"
    → last node before next heading is bulletList "sysdpuy1zl"
    → anchor_id = "sysdpuy1zl"

    LEAVE anchor_id null when:
    - operation_type is "create" → whole document is new
    - operation_type is "append" → goes at the very end
    - operation_type is "prepend" → goes at the very start

    FALLBACK:
    - If you cannot confidently identify the correct node → use "append" with anchor_id null

    ─────────────────────────────────────────────────────────────────────────────
    PLANNING RULES:
    ─────────────────────────────────────────────────────────────────────────────
    - Read the full request AND the doc_json before deciding operation_type
    - doc_json empty → operation_type is almost always "create"
    - doc_json not empty + content goes at end → "append"
    - doc_json not empty + content goes at start → "prepend"
    - doc_json not empty + user names a specific section to insert after → "insert" + anchor_id
    - doc_json not empty + user wants to modify existing text → "replace"
    - "write a conclusion" with doc → append (not create)
    - "write a conclusion" without doc → create
    - "rephrase the conclusion" → edit + replace (not write)
    - "add a methodology after the introduction" → write + insert + anchor_id from introduction node
    - "remove AI from this essay" → humanize + replace
    - "paraphrase this so it's not plagiarism" → deplagiarize + replace
    - "make a flowchart of this process" → diagram + create
    - For write tasks needing 3+ sections always produce an outline
    - For all other tasks leave outline empty'''),
    ("user", '''Request: {rephrased_query}
    Document: {doc_json}
    Plan:''')
])

planner_chain = planner_prompt | model_with_structured_output

