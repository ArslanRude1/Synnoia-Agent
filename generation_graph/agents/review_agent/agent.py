from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal, Optional
import os

load_dotenv()

class ReviewResponse(BaseModel):
    issues_found: bool = Field(default=False, description="Whether issues were found")
    issues: list[str] = Field(default=[], description="The issues found")

model = ChatOpenAI(
    model="gpt-5.4",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7,
    max_retries=2,
    reasoning_effort="high"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", '''You are a Generation Reviewer. You receive the output from any generation agent and validate it before it reaches the frontend.

    For document nodes (write, edit, humanize, deplagiarize tasks) check:
    - No node uses "type": "doc" or "content" arrays — must use "nodes" array
    - Every ParagraphNode has a non-empty segments list
    - HeadingNode has a text field, not a content array
    - No invented fields outside the schema
    - operation_type matches the request — "add conclusion" must be "append" not "create"
    - Content addresses the original request — wrong tone, missing sections, or off-topic content

    For diagram tasks check:
    - diagram_type is one of the valid types
    - All edges reference valid node ids
    - mermaid_syntax is present and syntactically correct
    - Labels are concise

    If issues found → return issues_found as True and issues as a list of strings
    If no issues → return issues_found as False and issues as an empty list'''),
    ("user", '''Original request: {rephrased_query}
    Task type: {task_type}
    Operation type: {operation_type}

    Generated output: {generated_output}

    Reviewed output:''')
])

review_chain = prompt | model.with_structured_output(ReviewResponse)

