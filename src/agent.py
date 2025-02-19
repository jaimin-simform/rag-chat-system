from __future__ import annotations

from dataclasses import dataclass
from dotenv import load_dotenv
from litellm import AsyncOpenAI
import logfire
import asyncio
import httpx
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from typing import List
from chromadb.api.models.Collection import Collection


env_path = os.path.join(os.path.dirname(__file__), '../config/.env')
load_dotenv(env_path)

llm = os.getenv("LLM_MODEL", "gpt-3.5-turbo-0125")
model = OpenAIModel(llm)

logfire.configure(send_to_logfire="if-token-present")



@dataclass
class PydanticAIDeps:
    collection: Collection
    openai_client: AsyncOpenAI


system_prompt = """
You work for a client. I will provide the client name and industry.
Review all knowledge documents thoroughly and provide detailed, evidence-backed answers aligned with the given behavioral challenge and choice drivers.

Your task is to:
1. Address the behavioral challenge based on processed content.
2. Prioritize responses according to the provided choice drivers.
3. Cite specific examples when possible.
4. Clarify assumptions and state when relevant information is unavailable.

Ensure responses align with the client name and industry context.

### Example Format:

**Choice Driver:**  
Job outcomes — e.g., high graduate employment rates, strong career support  

**Brand Proof Points:**  
**High Graduate Employability:**  
— [University] ranks #1 in Victoria for graduate employment.  
— Top 1% worldwide, with top rankings in disciplines like sport science (#1) and education (#5).  

**TALENT Career Services:**  
— **Award-Winning:** Australia’s #1 university career service (2017–2020).  
— **Comprehensive:** Career coaching, resume support, and success workshops.  
— **24/7 Online Tools:** Resume review and career resources available anytime.  

**Work-Ready Graduates:**  
— Practical, industry-oriented education for valued skills and experience.  
— Lifelong career support through [University] TALENT.

Ensure the final output follows this structured style.
"""

pydantic_ai_agent = Agent(
    model, system_prompt=system_prompt, deps_type=PydanticAIDeps, retries=2
)


async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536


@pydantic_ai_agent.tool
async def retrieve_relevant_documentation(
    ctx: RunContext[PydanticAIDeps], user_query: str
) -> str:
    """Retrieve relevant documentation chunks based on the query."""
    try:
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)

        results = ctx.deps.collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas"],
        )

        if not results["documents"][0]:
            return "No relevant documentation found."

        formatted_chunks = []
        for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
            chunk_text = f"""
# {metadata['title']}

{doc}

Source: {metadata['url']}
"""
            formatted_chunks.append(chunk_text)

        return "\n\n---\n\n".join(formatted_chunks)

    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"


@pydantic_ai_agent.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """Retrieve a list of all available documentation pages."""
    try:
        results = ctx.deps.collection.get(include=["metadatas"])

        if not results["metadatas"]:
            return []

        urls = sorted(set(meta["url"] for meta in results["metadatas"]))
        return urls

    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []


@pydantic_ai_agent.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """Retrieve the full content of a specific documentation page."""
    try:
        results = ctx.deps.collection.get(
            where={"url": url}, include=["documents", "metadatas"]
        )

        if not results["documents"]:
            return f"No content found for URL: {url}"

        sorted_results = sorted(
            zip(results["documents"], results["metadatas"]),
            key=lambda x: x[1]["chunk_number"],
        )

        page_title = sorted_results[0][1]["title"].split(" - ")[0]
        formatted_content = [f"# {page_title}\n"]

        for doc, _ in sorted_results:
            formatted_content.append(doc)

        return "\n\n".join(formatted_content)

    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"
