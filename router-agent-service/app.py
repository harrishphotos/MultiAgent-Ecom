"""
This module contains the router agent logic for intent recognition
and delegation to specialized agents in the multi-agent system.
"""

from datetime import datetime
from typing import Literal, List
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import httpx
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.llms.ollama import Ollama
from config import LLM_MODEL, OLLAMA_BASE_URL

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")


class RouterSchema(BaseModel):
    """Schema to represent user intent and order ID from customer queries."""

    intent: List[Literal["order_status", "refund_policy", "general"]] = Field(
        description="this is users primary intent as it can e any of refund_policy, order_status, general ",
        min_items=1,
    )
    order_id: str | None = Field(
        description="this is users order_id should be None if not found"
    )


def initialize_router_agent():
    """func than give a router chain to get llm output based on the context"""

    llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)

    app.state.output_phraser = JsonOutputParser(pydantic_object=RouterSchema)
    string_parser = StrOutputParser()

    router_prompt = ChatPromptTemplate.from_template(
        """You are an expert in understanding customer intent(s) based on their query.
            You MUST extract ALL intents that apply from the following list:
            ["order_status", "refund_policy", "general"]

            If the query contains an order ID matching 'ord' followed by digits (e.g., ord456),
            include "order_status" intent.

            if the query has some intention about returning or refund or policy relation 
            you should include the intent refund_policy.

            general is only for the querys that does not have any meaning related to "order_status" and "refund_policy"

            Your output MUST be ONLY a valid JSON object with these fields:
            - "intent": list of intents detected (at least one)
            - "order_id": string of the order ID found or null if none

        Output Instructions: 
        {output_instructions}
        
        User Query:
        {query}
        
        """
    )

    synthesizer_prompt = ChatPromptTemplate.from_template(
        """You are a polite customer support agent 
        who specializes in answering user Question based on the information gathered.
        Thinks as one and always ask yourself that "What answer can i give to customer's 
        question based on the context that i have"

    Instructions:    
        -Only provide answer based on the context that you have, say that you dont know if you dont 
        have the relevent information or supporting context.
        -Answers should be direct, short, polite.
        -no need to show how you came to the answer or the reasoning.
        -There may be return calculations use your calculation skills
        -IMPORTANT: your answer is final there shouldn't be any follow up questions

    context:
    {informations}

    current date(use if nessasory):
    {current_date}

    Customer Question:
    {query}


    """
    )
    app.state.router_agent = (
        router_prompt | llm | string_parser | app.state.output_phraser
    )
    app.state.response_synthesizer = synthesizer_prompt | llm


async def dynamic_router(query: str, request: Request):
    """A dynamic router that can handle multiple intends and synthesize a final answer based in information"""
    information = {}
    agent = request.app.state.router_agent
    output = await agent.ainvoke(
        {
            "query": query,
            "output_instructions": request.app.state.output_phraser.get_format_instructions(),
        }
    )
    intents = output["intent"]
    order_id = output["order_id"]
    logger.info("intents : %s", intents)
    current_date = datetime.now().strftime("%Y-%m-%d")
    for intent in intents:
        if intent == "order_status":
            if not order_id:
                information["order_status"] = (
                    "eventhough user asking about an order order id is not provided"
                )
                continue
            url = f"http://order-agent-service:8000/lookup/{order_id}"
            async with httpx.AsyncClient() as client:
                try:
                    res = await client.get(url)
                    res.raise_for_status()
                    data = res.json()
                    information["order_details"] = data.get(
                        "result", "unable to get order details"
                    )
                except Exception as e:
                    print(f"error occured {str(e)}")
        elif intent in ["refund_policy", "general"]:
            url = "http://policy-agent-service:8001/policy_query"

            async with httpx.AsyncClient() as client:
                try:
                    res = await client.post(url, json={"query": query})
                    res.raise_for_status()
                    data = res.json()
                    information["policy_context"] = data.get(
                        "policy_context", "unable to process policy info"
                    )
                except Exception as e:
                    print(f"unable to process policy info {str(e)}")
        else:
            information["error"] = "unable to process this information"
    synth = request.app.state.response_synthesizer
    final_output = await synth.ainvoke(
        {"query": query, "informations": information, "current_date": current_date}
    )
    return final_output


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
    response = await dynamic_router(query, request)
    return templates.TemplateResponse(
        "index.html", {"request": request, "response": response, "query": query}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
