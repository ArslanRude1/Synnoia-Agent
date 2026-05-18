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

class EditResponse(BaseModel):
    document: Document
    action_summary: str = Field(description="A summary of the edits made")

model = ChatOpenAI(
    model="gpt-5.4",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.5,
    max_retries=2,
    reasoning_effort="medium",
    streaming=True
)

model_with_structured_output = model.with_structured_output(EditResponse)

edit_prompt = ChatPromptTemplate([
    ("system", '''You are a Document Editor. You modify existing text according to the user's instruction and return the edited content as valid document JSON.

    RULES:
    - Operate ONLY on the provided text — do not add unrequested sections
    - Preserve all facts, names, dates, and figures from the original
    - Do not add disclaimers or meta-commentary about the edits made
    - Output ONLY the JSON — no markdown fences
    - After completing the task, write an action_summary describing what was edited
    - Scale the length to match the complexity:
    * Single operation on a short text (rephrase a sentence, fix grammar) → 1-2 sentences
    * Medium operation (expand a section, shorten a paragraph, change tone) → 2-3 sentences
    * Full document edit (restructure entire document, rewrite all sections) → 4-5 sentences
    - Always mention: what edit operation was performed, what text was affected, and what changed
    - Write in past tense, from the perspective of the editor reporting back to the user
    - Be specific — never write 'The text was edited'
    - Do not pad short edits — fixing grammar in one paragraph deserves one sentence'''),
    ("user", '''Instructions: {instructions}
    Reference document (mirror tone and style): {doc_json}
    Write:''')
])

edit_chain = edit_prompt | model_with_structured_output
