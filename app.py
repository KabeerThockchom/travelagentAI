from flask import Flask, request, jsonify, send_from_directory, Response, redirect, url_for, render_template, Blueprint
from flask_cors import CORS
from datetime import date
from typing import Optional
from duffel_api import Duffel
from duffel_api.http_client import ApiError
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
import os
import logging
from dotenv import load_dotenv
import langchain
import io


langchain.debug = True

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Duffel client
duffel_client = Duffel(access_token=os.getenv("DUFFEL_ACCESS_TOKEN"))

class SearchFlights(BaseModel):
    origin: str = Field(..., description="Origin airport code")
    destination: str = Field(..., description="Destination airport code")
    departure_date: str = Field(..., description="Departure date in YYYY-MM-DD format")
    return_date: Optional[str] = Field(None, description="Return date in YYYY-MM-DD format (optional)")
    cabin_class: Optional[str] = Field(None, description="Cabin class of the flight (economy, business, first)")


@tool
def search_flights(data: SearchFlights):
    """Search for flights with the given details."""
    client = duffel_client
    slices = [
        {"origin": data.origin, "destination": data.destination, "departure_date": data.departure_date},
    ]
    if data.return_date:
        slices.append({"origin": data.destination, "destination": data.origin, "departure_date": data.return_date})
    
    if data.cabin_class:
        offer_request = (
            client.offer_requests.create()
            .passengers([{"type": "adult"}])
            .slices(slices)
            .return_offers()
            .cabin_class(data.cabin_class)
            .execute()
        )
    else:
        offer_request = (
            client.offer_requests.create()
            .passengers([{"type": "adult"}])
            .slices(slices)
            .return_offers()
            .execute()
        )

    return [(offer.id, offer.owner.name, offer.slices[0].segments[0].departing_at, offer.total_amount, offer.total_currency) for offer in offer_request.offers]

class SelectOffer(BaseModel):
    offer_id: str = Field(..., description="Selected offer ID")

@tool
def select_offer(data: SelectOffer):
    """Select an offer to proceed with booking."""
    return data.offer_id

class BookFlight(BaseModel):
    id: str = Field(..., description="Selected offer ID")
    given_name: str = Field(..., description="Passenger's given name")
    family_name: str = Field(..., description="Passenger's family name")
    born_on: str = Field(..., description="Passenger's date of birth")
    title: str = Field(..., description="Passenger's title")
    gender: str = Field(..., description="Passenger's gender (m for male, f for female)")
    phone_number: str = Field(..., description="Passenger's phone number")
    email: str = Field(..., description="Passenger's email")

@tool
def book_flight(data: BookFlight):
    """Put a flight booking on hold with the given offer and passenger details."""
    client = duffel_client
    try:
        # Retrieve the offer details
        offer = client.offers.get(data.id)
        
        if not offer:
            return f"Offer with ID {data.id} not found."

        passengers = [
            {
                "phone_number": data.phone_number,
                "email": data.email,
                "title": data.title,
                "gender": data.gender,
                "family_name": data.family_name,
                "given_name": data.given_name,
                "born_on": data.born_on,
                "id": offer.passengers[0].id,
            }
        ]

        order = (
            client.orders.create()
            .passengers(passengers)
            .selected_offers([data.id])
            .hold()
            .execute()
        )

        return f"Created hold order {order.id} with booking reference {order.booking_reference}"
    except ApiError as e:
        return f"Failed to put flight on hold: {e}"

from duffel_api.api.booking.payments import PaymentClient

class CreatePayment(BaseModel):
    order_id: str = Field(..., description="Order ID for the held booking")
    amount: str = Field(..., description="Payment amount")
    currency: str = Field(..., description="Payment currency")
    payment_type: str = Field(..., description="Payment type (e.g., 'balance')")

@tool
def create_payment(data: CreatePayment):
    """Create a payment for the held booking."""
    client = PaymentClient(access_token=os.getenv("DUFFEL_ACCESS_TOKEN"))
    try:
        payment = {
            "amount": data.amount,
            "currency": data.currency,
            "type": data.payment_type
        }
        payment_response = (
            client.create()
            .order(data.order_id)
            .payment(payment)
            .execute()
        )
        return f"Payment created with ID {payment_response.id}"
    except PaymentClient.InvalidPayment as e:
        return f"Invalid payment data: {e}"
    except PaymentClient.InvalidPaymentType as e:
        return f"Invalid payment type: {e}"
    except ApiError as e:
        return f"Failed to create payment: {e}"

class CancelFlight(BaseModel):
    order_id: str = Field(..., description="Order ID to cancel")

@tool
def cancel_flight(data: CancelFlight):
    """Cancel a flight with the given order ID."""
    client = duffel_client
    try:
        order_cancellation = client.order_cancellations.create(data.order_id)
        client.order_cancellations.confirm(order_cancellation.id)
        return f"Order {data.order_id} has been canceled. Refund amount: {order_cancellation.refund_amount} {order_cancellation.refund_currency}"
    except ApiError as e:
        logger.error(f"Failed to cancel flight: {e}")
        return f"Failed to cancel flight: {e}"

# Define the chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system",f"""
    You are a flight booking assistant powered, designed to assist users with the following specific tasks:

    1. Search for flights: You can search for available flights based on the user's provided origin, destination, departure date, and optional return date.

    2. Select flight offers: After searching for flights, you can help users select a specific flight offer to proceed with the booking process.

    3. Book flights: Once a flight offer is selected, you can assist users in booking the flight by collecting necessary passenger details such as name, date of birth, title, gender, phone number, and email.

    4. Create payments: You can create payments for the booked flights using the order ID, payment amount, currency, and payment type (e.g., 'balance').

    5. Cancel flights: If needed, you can help users cancel their booked flights using the order ID.

    Please note that today's date is {date.today()}. Your role is strictly limited to assisting with these flight booking tasks. You should not engage in any conversations or tasks unrelated to flight booking.

    If a user asks for assistance with anything outside the scope of these defined tasks, politely inform them that you are a specialized flight booking assistant and cannot help with other matters. Respond with something along the lines of:

    "I apologize, but I am a specialized flight booking assistant. My capabilities are limited to helping with flight searches, selecting flight offers, booking flights, creating payments, and canceling flights. I cannot assist with any tasks or conversations outside of these defined responsibilities. If you have any flight-related questions, please let me know, and I'll do my best to help."
    """),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Bind the tools to the model
tools = [search_flights, select_offer, book_flight, create_payment, cancel_flight]

# Initialize the LLM with the tools
llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) #return_intermediate_steps = True

app = Flask(__name__, static_folder='./build', static_url_path='/')
@app.route('/')
def serve_react():
    return send_from_directory(app.static_folder, 'index.html')


CORS(app)

# Initialize an empty chat history list
chat_history = []

# Function to calculate the number of tokens in a message
def count_tokens(message):
    return len(message.split()) * 4  # Approximation based on the assumption that 1 token = 4 characters

# Function to manage chat history
def manage_chat_history(chat_history, max_tokens=100000):
    total_tokens = sum(count_tokens(msg.content) for msg in chat_history)
    
    while total_tokens > max_tokens:
        removed_message = chat_history.pop(0)
        total_tokens -= count_tokens(removed_message.content)

@app.route('/chat', methods=['POST'])
def chat():
    global chat_history

    user_input = request.json.get("input")
    
    # Add user input to the chat history
    chat_history.append(HumanMessage(content=user_input))

    # Manage the chat history to ensure it doesn't exceed the token limit
    manage_chat_history(chat_history, max_tokens=100000)

    # Invoke the agent with the updated chat history
    response = agent_executor.invoke(
        {
            "input": user_input,
            "chat_history": chat_history,
        }
    )

    # Add agent response to the chat history
    chat_history.append(AIMessage(content=response["output"]))

    # Manage the chat history to ensure it doesn't exceed the token limit
    manage_chat_history(chat_history, max_tokens=100000)

    return jsonify({"response": response["output"]})

if __name__ == '__main__':
    app.run(debug=True)