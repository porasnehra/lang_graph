import os
import requests
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic import hub

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
cse_id = os.getenv("GOOGLE_CSE_ID")
weather_key = os.getenv("OPENWEATHER_API_KEY")


@tool
def get_mandi_price(query: str) -> str:
    search_url = f"https://www.googleapis.com/customsearch/v1?key={google_api_key}&cx={cse_id}&q={query}"

    try:
        search_res = requests.get(search_url).json()
        target_url = search_res["items"][0]["link"]

        tables = pd.read_html(target_url)
        if not tables:
            return "No price tables found on the target website."

        df = tables[0]
        summary = df.head(5).to_string(index=False)
        return f"Source: {target_url}\n\nLatest Mandi Rates:\n{summary}"

    except Exception as e:
        return f"Could not retrieve Mandi data: {str(e)}"


@tool
def weather(location: str) -> str:
    """Gets current weather for a specific location."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={weather_key}&units=metric"
    data = requests.get(url).json()
    if "main" in data:
        return f"Weather in {location}: {data['main']['temp']}°C, {data['weather'][0]['description']}."
    return "Weather info unavailable."


tools = [get_mandi_price, weather]
model = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", api_key=google_api_key)
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

if __name__ == "__main__":
    query = "What is the price of Mustard in Yamunanagar today?"
    result = agent_executor.invoke({"input": query})
    print("\nFinal Answer:", result["output"])
