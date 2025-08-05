# The "manager" that decides which agent to use 
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
    """ this is our llm output structure """
    intent: Literal["order_status", "refund_policy", "general"] = Field(
        description="The user's primary intent. Is it about an order status, a refund, or something else?"
    )
    order_id: str | None = Field(
        description="The order ID mentioned by the user. Should be null if not found."
    )

llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)

output_parser = JsonOutputParser(pydantic_object=RouterSchema)

router_prompt = ChatPromptTemplate.from_template(
    """You are an expert at understanding customer queries.
    Analyze the user's input and determine their intent and any mentioned order ID.
    
    IMPORTANT: Respond with ONLY valid JSON. Do not include any explanations or additional text.
    
    {format_instructions}

    User's query:
    {query}
    
    JSON Response:"""
)

router_chain = router_prompt | llm | output_parser

async def route_query(query: str) -> str:
    """ using llm router chain to understand the intent to call the correct agent"""

    # Use async LangChain method - must work, no fallbacks
    router_result = await router_chain.ainvoke({
        "query": query,
        "format_instructions": output_parser.get_format_instructions()
    })
    intent = router_result['intent']
    order_id = router_result['order_id']

    if intent == "order_status":
        if not order_id:
            return "I can help with that could you please provide me with the order id ?"
        else:
            try:
                api_url = f"http://order-agent-service:8000/lookup/{order_id}"
                async with httpx.AsyncClient() as client:
                    response = await client.get(api_url)
                    response.raise_for_status()
                    return response.json().get('result', 'Error fetching order details.')
            except httpx.RequestError:
                return "Sorry, I'm having trouble connecting to the order system right now."

    
    elif intent in ["refund_policy", "general"]:
        try:
            api_url = "http://policy-agent-service:8001/policy_query"
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0, read=50.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
            ) as client:
                response = await client.post(api_url, json={"query": query})
                response.raise_for_status()
                result = response.json()
                return result.get('answer', 'Error fetching policy.')
        except httpx.RequestError:
            return "Sorry, I'm having trouble connecting to the policy system right now."
        except httpx.HTTPStatusError:
            return "Sorry, there was an error processing your policy request."
        except Exception:
            return "Sorry, I'm having trouble connecting to the policy system right now."
    
    else:
        return "I'm sorry, I'm not sure how to handle that request."

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "response": None, "query": None})

@app.post("/", response_class=HTMLResponse)
async def index_post(request: Request, query: str = Form(...)):
    response = await route_query(query)
    return templates.TemplateResponse("index.html", {"request": request, "response": response, "query": query})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
 