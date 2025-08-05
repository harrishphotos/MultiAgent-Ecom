# The logic for the agent that checks order status 
from fastapi import FastAPI
app = FastAPI()


ORDERS_DB = {
    "ORD123": {
        "item": "Blue T-Shirt",
        "status": "Processing",
        "ordered_date": "2025-07-26",
        "expected_delivery": "2025-07-29"
    },
    "ORD456": {
        "item": "Black Jeans",
        "status": "Delivered",
        "ordered_date": "2025-07-20",
        "expected_delivery": "2025-07-23"
    },
    "ORD789": {
        "item": "Red Shoes",
        "status": "Cancelled",
        "ordered_date": "2025-07-25",
        "expected_delivery": None
    }
}

def look_up_order( order_id: str) -> str:
    """ creating a function to lookup details of ORDER ID"""
    order_details = ORDERS_DB.get(order_id.upper())

    if order_details:
        response = (
            f"Status for order {order_id.upper()} ({order_details['item']})\n"
            f"  -Status: {order_details['status']}\n"
            f"  -Ordered On: {order_details['ordered_date']}"
            )
        
        if order_details['status'] == "Processing":
            response += f"\n  -Expected Delivery: {order_details['expected_delivery']}"

        return response
    return f"Sorry, I could not find details on Order ID '{order_id}'."


@app.get("/lookup/{order_id}")
async def lookup_order_api(order_id: str):
    result = look_up_order(order_id)
    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


