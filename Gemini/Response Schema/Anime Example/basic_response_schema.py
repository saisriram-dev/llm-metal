import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from anime_basemodel import Anime

load_dotenv()
client = genai.Client(api_key=os.getenv("llm-metal-gemini"))


def llm_call(prompt, system=None, schema=None, model="gemini-2.5-flash"):
    # It is used to set the system instructions
    config = types.GenerateContentConfig(system_instruction=system)

    if schema is not None:
        # The below forces the output to be in JSON format
        config.response_mime_type = "application/json"

        # The below line forces the output JSON to match the format of our schema(basemodel)
        config.response_schema = schema

    res = client.models.generate_content(model=model, contents=prompt, config=config)

    # res.parsed hand us back a dictionary rather than a JSON string
    return res.parsed if schema is not None else res.text


out = llm_call("Give me info about Demon Slayer", "You are a anime addict", Anime)

# Now 'out' is an Anime object which follows the structure of the Anime Basemodel
print(out.name, out.status, out.seasons, out.episodes)
print(type(out))
