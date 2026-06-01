import json
from typing import Any

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek

load_dotenv()
load_dotenv("/app/.env")

MAX_STEPS = 5


def resolve_city(city: str) -> dict[str, Any]:
    response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": city,
            "count": 1,
            "language": "zh",
            "format": "json",
        },
        timeout=20,
    )
    response.raise_for_status()

    results = response.json().get("results", [])
    if not results:
        raise ValueError(f"找不到城市：{city}")

    return results[0]


def fetch_weather(city: str) -> dict[str, Any]:
    location = resolve_city(city)

    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "precipitation,"
                "wind_speed_10m"
            ),
            "timezone": location.get("timezone", "auto"),
        },
        timeout=20,
    )
    response.raise_for_status()

    return {
        "city": city,
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "current": response.json().get("current", {}),
    }


@tool
def get_weather_batch(cities: list[str]) -> str:
    """查询一个或多个城市的当前天气，返回温度、湿度、降水和风速。适合查询、比较和筛选城市天气。"""
    records = [fetch_weather(city) for city in cities]
    return json.dumps(records, ensure_ascii=False)


TOOLS = [get_weather_batch]
TOOL_MAP = {item.name: item for item in TOOLS}

llm = ChatDeepSeek(
    model="deepseek-v4-pro",
    temperature=0,
)

llm_with_tools = llm.bind_tools(TOOLS)


def run_agent(user_input: str) -> str:
    messages = [
        SystemMessage(
            content=(
                "你是天气助手。需要实时天气时必须调用工具。"
                "只能根据工具返回的数据回答，不要猜测天气状况。"
            )
        ),
        HumanMessage(content=user_input),
    ]

    for step in range(MAX_STEPS):
        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)

        print(f"\nStep {step + 1}:")
        print("Tool calls:")
        print(json.dumps(ai_message.tool_calls, ensure_ascii=False, indent=2))

        if not ai_message.tool_calls:
            return str(ai_message.content)

        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]

            if tool_name not in TOOL_MAP:
                raise ValueError(f"Unknown tool: {tool_name}")

            tool_message = TOOL_MAP[tool_name].invoke(tool_call)
            messages.append(tool_message)

            print("\nTool result:")
            print(tool_message.content)

    raise RuntimeError("Agent exceeded maximum steps")


def main():
    questions = [
        "上海现在天气怎么样？",
        "成都、北京、上海哪个城市现在温度最高？",
        "北京和上海温差是多少？",
        "成都、北京、上海哪些城市现在正在下雨？",
    ]

    for question in questions:
        print("=" * 80)
        print("User:", question)
        print("\nFinal answer:")
        print(run_agent(question))
        print()


if __name__ == "__main__":
    main()
