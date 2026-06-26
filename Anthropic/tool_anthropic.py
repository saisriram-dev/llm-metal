import os
import json
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("llm_metal_key"))

def get_weather(city: str) -> dict:
    # Pretend this hits a real weather API. Just faking it for learning
    return {"city": city, "temp_c": 31, "condition": "humid, partly cloudy"}

tools = [
    {   
        # name tells what function(in the code) to run when the tool is used
        "name": "get_weather",

        # Description is used by model to know when to use this tool
        "description": "Get the current weather for a city. "
                       "Use this whenever the user asks about the weather.",

        # Tells what information the function needs to the model
        "input_schema": {

            # Type=object tells that you will get a key-value pair
            "type": "object",

            # Here there's one field, city, and its type is "string" (For the function get_weather)
            # If your function also needed a number of days, you'd add "days": {"type": "number"} 
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, like 'Coimbatore'"
                }
            },
            "required": ["city"]
        }
    }
]

messages = [
    {"role": "user", "content": "What's the weather in Coimbatore?"}
]

while True:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024, # This is mandatory
        tools=tools,
        messages=messages,
    )

    messages.append({"role": "assistant", "content": response.content})

    # There will be a stop_reason for each response
    # BRANCH on stop_reason. This is the whole game.
    if response.stop_reason != "tool_use":
        # The model gave a final text answer. Pull the text block out and stop.
        """
        'next' is simply replacement for:

        for b in response.content:
            if b.type == "text":
                final_text = b.text
                break          # grab the first text block, stop looking
        """
        final_text = next(b.text for b in response.content if b.type == "text")

        print(final_text)
        break
    
    # Otherwise stop_reason == "tool_use": the model is asking us to run something.
    # response.content is a LIST of blocks — there may be several tool_use blocks.

    tool_results = []

    for block in response.content:
        if block.type == "tool_use":
            # block.name  -> which tool        ("get_weather")
            # block.input -> the arguments     ({"city": "Coimbatore"})
            # block.id    -> the call's unique id (we MUST echo it back)
            if block.name == "get_weather":
                """
                    get_weather(**{"city": "Coimbatore"}) is same as get_weather(city="Coimbatore")
                """
                result = get_weather(**block.input)
            else:
                result = {"error": f"unknown tool: {block.name}"}

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,              # links our answer to the model's request
                "content": json.dumps(result),        # the result, as a string
            })
    
    # Send the results back as a USER message (results are input flowing INTO the model).
    messages.append({"role": "user", "content": tool_results})
    # Loop repeats: the model now sees the weather data and writes the final answer.
