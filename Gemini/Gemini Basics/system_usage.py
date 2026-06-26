import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("llm-metal-gemini"))

response = client.models.generate_content(
    # The model which we are using
    model="gemini-2.5-flash",

    # The user query
    contents="Explain what an API does in 2 sentences.",

    # Used to define the system prompt
    config=types.GenerateContentConfig(
        system_instruction="You are terse. Never exceed 2 sentences"
    ),
)

print(response.text)
print(response.usage_metadata.total_token_count)
print(response.usage_metadata)

"""
An API (Application Programming Interface) allows different software applications to communicate 
and exchange informationwith each other. It defines the methods and data formats that applications 
can use to request services or share data, enabling them to work together seamlessly.

The indentation may be different but the text will be approximately same except for the token count:

cache_tokens_details=None cached_content_token_count=None candidates_token_count=45 
candidates_tokens_details=None prompt_token_count=21 prompt_tokens_details=[ModalityTokenCount(
  modality=<MediaModality.TEXT: 'TEXT'>,
  token_count=21
)] 
thoughts_token_count=68 tool_use_prompt_token_count=None tool_use_prompt_tokens_details=None 
total_token_count=134 traffic_type=None

prompt_token_count == Tokens in our prompt
candidates_token_count == Tokens in the output
thoughts_token_count == Thinking tokens used by the model
"""
