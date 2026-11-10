import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from a .env file in the project root (if present)
load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")
if not api_key:
  raise ValueError("NVIDIA_API_KEY environment variable not set. Add it to a .env file or export it in your shell.")

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = api_key
)


completion = client.chat.completions.create(
  model="nvidia/nemotron-3.5-super-vl-120b-a12b",
  messages=[{"role": "user", "content": "Hello Groq! confirm that you are working"}],
  temperature=0.0,
  top_p=0.9,
  max_tokens=16384,

  parallel_tool_calls=True,



  seed=42,
  extra_body={"chat_template_kwargs": {"enable_thinking": True}},
  stream=True
)

for chunk in completion:
  if not chunk.choices:
    continue
  reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
  if reasoning:
    print(reasoning, end="")
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")