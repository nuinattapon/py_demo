import os
import json
import requests
from typing import List, Dict
from together import Together
from dotenv import load_dotenv

def generate_with_multiple_input(
    messages: List[Dict],
    top_p: float = 1.0,
    temperature: float = 1.0,
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
        
    if together_api_key is None:
        together_api_key = os.getenv("TOGETHER_API_KEY", "")
        if together_api_key=="":
            raise Exception(
                "There is no TOGETHER_API_KEY"
            )        
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
