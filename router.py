# The "manager" that decides which agent to use 

from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.llms.ollama import Ollama


from agents.order_agent import look_up_order
from agents.policy_agent import create_policy_agent_chain
from config import LLM_MODEL



class RouterSchema(BaseModel):
    """ this is our llm output structure """
    intent: Literal["order_status", "refund_policy", "general"] = Field(
        description="The user's primary intent. Is it about an order status, a refund, or something else?"
    )
    order_id: str | None = Field(
        description="The order ID mentioned by the user. Should be null if not found."
    )

llm = Ollama(model=LLM_MODEL)

output_parser = JsonOutputParser(pydantic_object=RouterSchema)

router_prompt = ChatPromptTemplate.from_template(
    """You are an expert at understanding customer queries.
    Analyze the user's input and determine their intent and any mentioned order ID.
    
    {format_instructions}

    User's query:
    {query}
    """
)

router_chain = router_prompt | llm | output_parser

policy_agent = create_policy_agent_chain()

def route_query(query: str) -> str:
    """ using llm router chain to understand the intent to call the correct agent"""
    print(f"Router: Analyzing query '{query}' with LLM...")

    router_result = router_chain.invoke({
        "query" : query,
        "format_instructions" : output_parser.get_format_instructions()
    })

    print(f"Router: LLM analysis complete. Intent is '{router_result['intent']}'.")

    intent = router_result['intent']
    order_id = router_result['order_id']

    if intent == "order_status":
        if not order_id:
            print("Router: Intent is order_status, but no ID found. Asking user for ID.")
            return "I can help with that could you please provide me with the order id ?"
        else:
            return look_up_order(order_id)
    
    elif intent in ["refund_policy", "general"]:
        print("Router: Calling Policy Agent for a general or refund query.")
        response = policy_agent.invoke({"input": query})
        return response["answer"]
    
    else:
        return "I'm sorry, I'm not sure how to handle that request."
