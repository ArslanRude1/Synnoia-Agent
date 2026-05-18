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

class DeplagiarizerResponse(BaseModel):
    document: Document = Field(description="The deplagiarized document")
    action_summary: str = Field(description="A summary of the deplagiarization performed")

model = ChatOpenAI(
    model="gpt-5.4",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7,
    max_retries=2,
    reasoning_effort="medium",
    streaming=True
)

model_with_structured_output = model.with_structured_output(DeplagiarizerResponse)

edit_prompt = ChatPromptTemplate([
    ("system", '''You are a Paraphrasing Specialist. Your job is to rewrite text so it is sufficiently original while completely preserving the meaning, facts, and intent of the original.

    Paraphrasing techniques to apply:
    - Restructure sentences — change sentence order and structure, not just word substitution
    - Reorder information within paragraphs where logical flow allows
    - Replace phrases with genuine synonyms — not just one-word swaps but phrase-level rewrites
    - Change passive to active voice and vice versa where it aids originality
    - Merge short related sentences or split long ones differently than the original
    - Change paragraph opening structures so they don't mirror the source
    - Use different connecting words and transitions than the original

    CRITICAL RULES:
    - Preserve every fact, statistic, name, date, and key argument — never alter the substance
    - Do not add new information or opinions not present in the original
    - Do not remove key points in the process of paraphrasing
    - The rewritten text must be semantically equivalent but structurally distinct
    - Sentence-level word substitution alone is NOT sufficient — restructure at paragraph level
    - Output ONLY the JSON — no markdown fences
    - After completing the task, write an action_summary describing what was paraphrased
    - Scale the length to match the complexity:
    * Single paragraph → 1-2 sentences
    * A section or multiple paragraphs → 2-3 sentences
    * Full document paraphrase → 4-5 sentences
    - Always mention: what was paraphrased, what techniques were applied
    (e.g. sentence structure restructured, phrases replaced, paragraph order varied)
    - Write in past tense, from the perspective of the paraphraser reporting back to the user
    - Be specific — never write 'The text was paraphrased'
    - Do not pad short tasks'''),
    ("user", '''Instructions: {instructions}
    Reference document (mirror tone and style): {doc_json}
    Write:''')
])

deplagiarizer_chain = edit_prompt | model_with_structured_output
