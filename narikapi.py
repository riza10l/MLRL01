import os
# pyrefly: ignore [missing-import]
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
)

response = client.responses.create(
    model="qwen3.6-plus",
    input="Which is larger, 9.9 or 9.11?",
    extra_body={
        "enable_thinking": True  # Enable thinking mode
    }
)

# Iterate through output items
for item in response.output:
    if item.type == "reasoning":
        # Print reasoning summary
        print("[Reasoning]")
        for summary in item.summary:
            print(summary.text[:500])  # Truncate to first 500 chars
        print()
    elif item.type == "message":
        # Print final answer
        print("[Answer]")
        print(item.content[0].text)