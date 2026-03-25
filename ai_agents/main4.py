import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic import hub

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
weather_key = os.getenv("OPENWEATHER_API_KEY")

ddg_search = DuckDuckGoSearchRun()


@tool
def search(query: str) -> str:
    return ddg_search.run(query)


@tool
def weather(location: str) -> str:
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={weather_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return f"The temperature in {location} is {data['main']['temp']}°C with {data['weather'][0]['description']}."
    return f"Error fetching weather for {location}."


@tool
def get_mandi_price(crop: str, state: str = "Haryana") -> str:
    
    search_query = f"latest {crop} mandi price today in {state} district wise"
    search_results = ddg_search.run(search_query)

    return f"Market Data found for {crop} in {state}: {search_results[:1000]}"


tools = [search, weather, get_mandi_price]

model = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    api_key=google_api_key,
    max_retries=0,
)

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

if __name__ == "__main__":
    query = "What is the latest mandi price of Mustard in Yamunanagar, Haryana? Also, tell me if it will rain there today."
    response = agent_executor.invoke({"input": query})
    print(response["output"])
