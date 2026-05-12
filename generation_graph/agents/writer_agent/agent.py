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

from schema.document_schema import Document

class WriterResponse(BaseModel):
    document: Document
    action_summary: str = Field(description="A summary of the document written")

model = ChatOpenAI(
    model="gpt-5.4",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7,
    max_retries=2,
    reasoning_effort="medium"
)

model_with_structured_output = model.with_structured_output(WriterResponse)

prompt = ChatPromptTemplate.from_messages([
    ("system", '''You are a Document Writer. You write clean, well-structured content and return valid document JSON.

    RULES:
    - Never use "type": "doc" or "content" arrays
    - Every ParagraphNode must have non-empty segments
    - Mirror tone and style of reference document if provided
    - Write only what is instructed — no extra sections or filler
    - Output ONLY the JSON — no markdown fences
    - After completing the task, write an action_summary describing what was written
    - Scale the length to match the complexity:
    * Single section (conclusion, introduction, abstract) → 1-2 sentences
    * Medium task (write a blog, write an essay, write a report section) → 3-4 sentences
    * Full multi-section document (report, research paper, assignment) → 5-6 sentences
    - Always mention: what was written, the structure used, topics covered, and tone applied
    - Write in past tense, from the perspective of the writer reporting back to the user
    - Be specific — never write 'The document was created successfully'
    - Do not pad short tasks — a single conclusion section deserves one sentence'''),
    ("user", '''Instructions: {instructions}
    Outline (follow this section plan if provided, empty means write directly from instructions): {write_outline}
    Reference document (mirror tone and style): {doc_json}
    Write:''')
])

writer_chain = prompt | model_with_structured_output
