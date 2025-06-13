import gradio as gr
from faq_bot import chat_with_faq  # reuse the function from your backend

def chat_interface(message, history):
    response = chat_with_faq(message)
    return response

gr.ChatInterface(fn=chat_interface, type="messages").launch()
