"""
TogetherAI integration
"""

import json
import os
from typing import List, Dict
import requests
from together import Together

from dotenv import load_dotenv


def generate_with_single_input(
    prompt: str,
    role: str = "user",
    top_p: float = 1.0,
    temperature: float = 1.0,
    max_tokens: int = 500,
    model: str = "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    together_api_key=None,
    **kwargs,
):

    if top_p is None:
        top_p = "none"
    if temperature is None:
        temperature = "none"

    payload = {
        "model": model,
        "messages": [{"role": role, "content": prompt}],
        "top_p": top_p,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **kwargs,
    }

    if (not together_api_key) and ("TOGETHER_API_KEY" not in os.environ):
        url = os.path.join("https://api.together.xyz", "v1/chat/completions")
        response = requests.post(url, json=payload, verify=False)
        if not response.ok:
            raise Exception(f"Error while calling LLM: f{response.text}")
        try:
            json_dict = json.loads(response.text)
        except Exception as e:
            raise Exception(
                f"Failed to get correct output from LLM call.\nException: {e}\nResponse: {response.text}"
            )
    else:
        if together_api_key is None:
            together_api_key = os.getenv("TOGETHER_API_KEY", "")
        client = Together(api_key=together_api_key)
        json_dict = client.chat.completions.create(**payload).model_dump()
        json_dict["choices"][-1]["message"]["role"] = json_dict["choices"][-1][
            "message"
        ]["role"].name.lower()
    try:
        output_dict = {
            "role": json_dict["choices"][-1]["message"]["role"],
            "content": json_dict["choices"][-1]["message"]["content"],
        }
    except Exception as e:
        raise Exception(
            f"Failed to get correct output dict. Please try again. Error: {e}"
        )
    return output_dict


def generate_with_multiple_input(
    messages: List[Dict],
    top_p: float = None,
    temperature: float = None,
    max_tokens: int = 500,
    model: str = "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    together_api_key=None,
    **kwargs,
):
    payload = {
        "model": model,
        "messages": messages,
        "top_p": top_p,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **kwargs,
    }
    if (not together_api_key) and ("TOGETHER_API_KEY" not in os.environ):
        url = os.path.join("https://api.together.xyz", "v1/chat/completions")
        response = requests.post(url, json=payload, verify=False)
        if not response.ok:
            raise Exception(f"Error while calling LLM: f{response.text}")
        try:
            json_dict = json.loads(response.text)
        except Exception as e:
            raise Exception(
                f"Failed to get correct output from LLM call.\nException: {e}\nResponse: {response.text}"
            )
    else:
        if together_api_key is None:
            together_api_key = os.getenv("TOGETHER_API_KEY", "")
        client = Together(api_key=together_api_key)
        json_dict = client.chat.completions.create(**payload).model_dump()
        json_dict["choices"][-1]["message"]["role"] = json_dict["choices"][-1][
            "message"
        ]["role"].name.lower()
    try:
        output_dict = {
            "role": json_dict["choices"][-1]["message"]["role"],
            "content": json_dict["choices"][-1]["message"]["content"],
        }
    except Exception as e:
        raise Exception(
            f"Failed to get correct output dict. Please try again. Error: {e}"
        )
    return output_dict


if __name__ == "__main__":
    load_dotenv()
    # Example usage of the Together AI chat interface
    TOGETHER_API_KEY: str | None = os.getenv("TOGETHER_API_KEY")
    PROMPT = "Explain AI to me like I am a college student"
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": PROMPT},
    ]
    answer = generate_with_multiple_input(messages)
    print(answer)  # Display the model's response

    # Example call
    output = generate_with_single_input(prompt="What is the capital of France?")

    print("Role:", output["role"])
    print("Content:", output["content"])
