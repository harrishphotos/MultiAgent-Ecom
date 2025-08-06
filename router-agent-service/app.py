"""
This module contains the router agent logic for intent recognition
and delegation to specialized agents in the multi-agent system.
"""

from typing import Literal
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import httpx
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.llms.ollama import Ollama
from config import LLM_MODEL, OLLAMA_BASE_URL

app = FastAPI()
templates = Jinja2Templates(directory="templates")


class RouterSchema(BaseModel):
    """Schema to represent user intent and order ID from customer queries."""

    intent: Literal["order_status", "refund_policy", "general"] = Field(
        description="this is users primary intent as it can "
        "be either refund_policy or order_status or general"
    )
    order_id: str | None = Field(
        description="this is users order id which has a format "
        "of ORDXXX and if not fould should be None"
    )


def initialize_router_agent():
    """func than give a router chain to get llm output based on the context"""

    llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)

    app.state.output_phraser = JsonOutputParser(pydantic_object=RouterSchema)

    router_prompt = ChatPromptTemplate.from_template(
        """ You are an expert in understanding customer querys and providing required output.
        IMPORTANT: Provide output in only valid JSON format

        Output Instructions: 
        {output_instructions}
        
        User Query:
        {query}
    """
    )

    app.state.router_agent = router_prompt | llm | app.state.output_phraser


async def route_query(query: str, request: Request) -> str:
    """A centralized router agent that orchastrate the whole thing.
    As it is kind of hard coded not ideal but for studying purposes"""
    agent = request.app.state.router_agent
    output = await agent.ainvoke(
        {
            "query": query,
            "output_instructions": request.app.state.output_phraser.get_format_instructions(),
        }
    )

    if output["intent"] == "order_status":
        if not output["order_id"]:
            return "I can help with that could you please provide me the OrderId ?"
        url = f"http://order-agent-service:8000/lookup/{output['order_id']}"
        async with httpx.AsyncClient() as client:
            try:
                res = await client.get(url)
                res.raise_for_status()
                return res.json().get("result", "sorry error fetching order details")
            except Exception as e:
                return f"unexpected error occured {str(e)}"
    elif output["intent"] in ["refund_policy", "general"]:
        try:
            url = "http://policy-agent-service:8001/policy_query"
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0, read=50.0)
            ) as client:
                result = await client.post(url, json={"query": query})
                result.raise_for_status()
                return result.json().get(
                    "answer", "sorry error fetching policy details"
                )
        except Exception as e:
            return f"unexpecter error : {str(e)}"
    else:
        return "sorry this request is out of scope"


@app.on_event("startup")
def on_startup():
    """creates a router agent when starting up"""
    initialize_router_agent()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """serves the HTML page"""
    return templates.TemplateResponse(
        "index.html", {"request": request, "response": None, "query": None}
    )


@app.post("/", response_class=HTMLResponse)
async def index_post(request: Request, query: str = Form(...)):
    """Handles the query and serves the page with query and answer"""
    response = await route_query(query, request)
    return templates.TemplateResponse(
        "index.html", {"request": request, "response": response, "query": query}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
