import gradio as gr
import os
from faq_bot import chat_with_faq  # reuse the function from your backend

def chat_interface(message, history):
    response = chat_with_faq(message)
    return response

port = int(os.environ.get("PORT", 5057))


gr.ChatInterface(fn=chat_interface, type="messages").launch(server_name="0.0.0.0", server_port=port)
