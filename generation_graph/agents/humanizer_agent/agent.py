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

class HumanizerResponse(BaseModel):
    document: Document = Field(description="The humanized document")
    action_summary: str = Field(description="A summary of the humanization performed")

model = ChatOpenAI(
    model="gpt-5.4",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7,
    max_retries=2,
    reasoning_effort="medium"
)

model_with_structured_output = model.with_structured_output(HumanizerResponse)

edit_prompt = ChatPromptTemplate([
    ("system", '''You are a Text Humanization Specialist. Your job is to rewrite AI-generated text so it reads naturally, authentically, and undetectably human while fully preserving the original meaning, facts, and intent.

    AI writing patterns to eliminate:
    - Overly structured phrasing: "It is important to note that...", "In conclusion,", "Furthermore,"
    - Robotic transitions: "Additionally,", "Moreover,", "In summary,"
    - Hedging phrases: "It is worth mentioning that...", "One could argue that..."
    - Passive overuse: prefer active voice where natural
    - Uniform sentence length: vary sentence rhythm — mix short punchy sentences with longer flowing ones
    - Vocabulary that is technically correct but unnaturally formal for the context
    - Symmetrical paragraph structure — humans don't write in perfectly balanced paragraphs

    Humanization techniques to apply:
    - Vary sentence length and rhythm naturally — this is the strongest signal
    - Use contractions where appropriate for the tone (don't, it's, they're)
    - Add subtle imperfections in structure — not errors, just natural human flow
    - Use concrete specific language over abstract general statements
    - Replace generic filler phrases with direct statements
    - Break overly long compound sentences into shorter natural ones
    - Start sentences with "And" or "But" occasionally — humans do this

    CRITICAL RULES:
    - Preserve every fact, figure, name, date, and key argument — do not change the substance
    - Match the appropriate register — academic humanization differs from blog humanization
    - Do not make it sound casual if the original is formal — humanize within the correct register
    - Do not add new content or remove key points
    - Output ONLY the JSON — no markdown fences
    - After completing the task, write an action_summary describing what was humanized
    - Scale the length to match the complexity:
    * Single paragraph or short text → 1-2 sentences
    * A section or multiple paragraphs → 2-3 sentences
    * Full document humanization → 4-5 sentences
    - Always mention: what AI patterns were removed, what techniques were applied
    (e.g. sentence rhythm varied, contractions added, transitions replaced, structure broken up)
    - Write in past tense, from the perspective of the humanizer reporting back to the user
    - Be specific — never write 'The text was humanized'
    - Do not pad short tasks'''),
    ("user", '''Instructions: {instructions}
    Reference document (mirror tone and style): {doc_json}
    Write:''')
])

humanizer_chain = edit_prompt | model_with_structured_output
