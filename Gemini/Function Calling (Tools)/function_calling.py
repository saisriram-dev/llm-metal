import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()  # load env vars from .env so the API key isn't hardcoded
client = genai.Client(api_key=os.getenv("llm-metal-gemini"))

MODEL = "gemini-2.5-flash"

def fake_weather_api(location: str) -> dict:
    """A fake weather API that returns a dictionary with weather information for a given location."""
    # In a real-world scenario, this function would make an API call to a weather service.
    # Here, we return a hardcoded response for demonstration purposes.
    fake_db = {
        "london": 18,
        "new york": 25,
        "coimbatore": 30,
        "tokyo": 22,
        "sydney": 20
    }

    temp = fake_db.get(location.lower(), 25) # Default Fallback temperature if location not found

    return {
        "location": location,
        "temperature": temp,
        "unit": "Celsius"
    }

# The description gemini sees when using a tool
weather_function_declaration = {
    "name": "fake_weather_api",
    "description":"Used when the user wants to know about the weather conditions of a particular city",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city for which the user wants to know the weather conditions"
            }
        },
        "required": ["location"],
    }
}

# wrap the function declaration in a Tool object, this is what gets passed to the model
tools = types.Tool(function_declarations=[weather_function_declaration])
# bundle the tool(s) into the generation config so every call in this session can use them
config = types.GenerateContentConfig(
    tools=[tools]
)

def run_convo(user_prompt):
    # start the conversation with the user's message
    contents = [
        types.Content(role="user", parts=[types.Part(text=user_prompt)])
    ]

    # first call - ask gemini what to do, it may respond with text or ask to call a function
    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=config
    )

    candidate_parts = response.candidates[0].content.parts
    function_call_part = None

    # check each part of the response to see if gemini wants to call a function
    for part in candidate_parts:
        if part.function_call:
            function_call_part = part.function_call
            break

    if function_call_part is None:
        # no function call, so gemini just answered directly - print it and stop here
        print("Gemini has responded: ")
        print(response.text)
        return  # bug fix: without this, the code below crashes since function_call_part is None

    # pull out the function name and args gemini decided to use
    fn_name = function_call_part.name
    fn_args = dict(function_call_part.args)

    print(f"Gemini picked function: {fn_name}")
    print(f"Gemini picked arguments: {fn_args}")

    # route to the right function based on what gemini asked for
    if fn_name == "fake_weather_api":
        result = fake_weather_api(**fn_args)
    else:
        raise ValueError(f"Unknown function requested: {fn_name}")
    
    # add gemini's function call turn to the conversation history
    contents.append(response.candidates[0].content)
    # now send the function's result back to gemini so it can use it in its final answer
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_function_response(
                    name=fn_name,
                    response={"result": result}
                )
            ]
        )
    )

    # second call - now gemini has the function result and can give a proper final answer
    final_resp = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=config,
    )

    print("\nFinal answer from Gemini:")
    print(final_resp.text)
 
 
if __name__ == "__main__":
    run_convo("What's the temperature in Coimbatore right now?")
