import os
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain.schema import Document
import json
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
      
endpoint = os.getenv("ENDPOINT_URL")
deployment = os.getenv("DEPLOYMENT_NAME")
embeddingModel = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
      
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

embedding = AzureOpenAIEmbeddings(
    azure_endpoint=endpoint,
    azure_ad_token_provider=token_provider,
    deployment=embeddingModel,
    openai_api_version="2024-12-01-preview",
)

# Load the structured FAQ JSON
with open("faq.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)
# Create documents from question + answer
docs = []
for item in faq_data:
    content = f"Q: {item['question']}\nA: {item['answer']}"
    docs.append(Document(page_content=content))

# Index with FAISS
vectorstore = FAISS.from_documents(docs, embedding)
vectorstore.save_local("faq_index")

vectorstore = FAISS.load_local("faq_index", embedding, allow_dangerous_deserialization=True)

def chat_with_faq(question: str):
    docs = vectorstore.similarity_search(question, k=2)
    context = "\n\n".join(doc.page_content for doc in docs)
    print("\nContext from FAQ snippets:\n", context)
    system_message = {
        "role": "system",
        "content": f"You are a helpful AI assistant created by the platform engineering team. If you are very unsure about your answer, then refer to the platform team. DO NOT tell the user to do any manual stuff using cli commands or in the azure portal that would require control plane contributor. Use the following FAQ snippets to help answer user questions:\n\n{context}"
    }
    
    messages = [
        system_message,
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
