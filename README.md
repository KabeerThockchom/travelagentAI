
# TravelAgentAI

Travel Agent AI that can search flights, book flights, cancel flight bookings using OpenAI, LangChain, and Duffel API.

## Installation

To install the required dependencies, run the following command:

pip install -r requirements.txt

## Configuration

1. Create a `.env` file at the root of your project folder.
2. Add the following environment variables to your `.env` file:

OPENAI_API_KEY=<your_openai_api_key>
DUFFEL_ACCESS_TOKEN=<your_duffel_access_token>


Replace `<your_openai_api_key>` and `<your_duffel_access_token>` with your actual OpenAI API key and Duffel access token respectively.

If you want to add tracing with LangSmith to monitor agent flows.

Add the following environment variables to your `.env` file:

LANGCHAIN_API_KEY= <your_langsmith_api_key>
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_PROJECT = <project_name>

Replace `<your_langsmith_api_key>` and `<project_name>` with your actual LangSmith API key and chosen project name respectively.

## Usage

To run the application, execute the following command:

python app.py

This README provides clear instructions on how to install, configure, and use your travel agent AI application. If you need further assistance, feel free to reach out!


This should make it easier for users to understand and follow the instructions in you
