import os
import base64
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
      
endpoint = os.getenv("ENDPOINT_URL", "https://nhqtest.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "o4-mini")
      
# Initialize Azure OpenAI client with Entra ID authentication
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    azure_endpoint=endpoint,
    azure_ad_token_provider=token_provider,
    api_version="2025-01-01-preview",
)



def load_faq(filename="faq.txt"):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

def create_system_message(faq_text):
    return {
        "role": "system",
        "content": f"You are a helpful AI assistant created by the platform engineering team to support developers. Use the following FAQ content to answer their questions clearly and concisely, helping them solve common issues before contacting the platform team directly.:\n{faq_text}"
    }

def chat_with_faq(question: str):
    faq_text = load_faq()
    messages = [
        create_system_message(faq_text),
        {"role": "user", "content": question}
    ]
    
    completion = client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_completion_tokens=1000
    )
    print("\nAnswer:", completion.choices[0].message.content)
    # Return the assistant's reply
    return completion.choices[0].message.content
