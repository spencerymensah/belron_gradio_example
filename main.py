import os
import time
import gradio as gr
import google.generativeai as genai
import pymupdf

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini and returns the file object and its URI.

    See https://ai.google.dev/gemini-api/docs/prompting_with_media
    """
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file, file.uri

def wait_for_files_active(file_uri):
    file_id = file_uri.split("/")[-1]
    print(f"Waiting for file to become active: {file_id}")
    file = genai.get_file(file_id)
    while file.state.name == "PROCESSING":
        time.sleep(10)
        file = genai.get_file(file_id)
    if file.state.name != "ACTIVE":
        raise Exception(f"File {file_id} failed to process")
    print(f"File {file_id} is active")

def add_message(history, message):
    files = message.get("files", [])
    print(files)
    text = message.get("text", "")
    if files:
        uploaded_files = []
        for file_path in files:
            file, file_uri = upload_to_gemini(file_path)
            wait_for_files_active(file_uri)
            uploaded_files.append(file)
        uploaded_files.append(text)
        parts = uploaded_files
    else:
        parts = [text]

    doc = pymupdf.open("example.pdf") 
    base_content = ""
    for page in doc:
        base_content += page.get_text()
    parts.append(base_content)
    print(parts)

    chat_history = [{"role": "user", "parts": parts}]
    chat_session = model.start_chat(history=chat_history)
    response = chat_session.send_message(text)
    
    if files:
        history.append((files, response.text))
    else:
        history.append((text, response.text))
    
    return history, gr.MultimodalTextbox(value=None, interactive=True)

with gr.Blocks(fill_height=True) as demo:
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        bubble_full_width=False,
        scale=1,
    )

    chat_input = gr.MultimodalTextbox(interactive=True,
                                      file_count="multiple",
                                      placeholder="Enter message or upload file...", show_label=False)

    chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
    chat_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

demo.queue()
demo.launch()
