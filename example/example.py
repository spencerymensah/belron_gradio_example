import os
import time
import pymupdf # imports the pymupdf library
import google.generativeai as genai

genai.configure(api_key="AIzaSyA32_OACE1eVlPVbFc0TkVijjxQFfuh8KM")

def upload_to_gemini(path, mime_type=None):
  """Uploads the given file to Gemini.

  See https://ai.google.dev/gemini-api/docs/prompting_with_media
  """
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file

def wait_for_files_active(files):
  """Waits for the given files to be active.

  Some files uploaded to the Gemini API need to be processed before they can be
  used as prompt inputs. The status can be seen by querying the file's "state"
  field.

  This implementation uses a simple blocking polling loop. Production code
  should probably employ a more sophisticated approach.
  """
  print("Waiting for file processing...")
  for name in (file.name for file in files):
    print(f"Waiting for {name} to be ready...")
    file = genai.get_file(name)
    while file.state.name == "PROCESSING":
      print(".", end="", flush=True)
      time.sleep(10)
      file = genai.get_file(name)
    if file.state.name != "ACTIVE":
      raise Exception(f"File {file.name} failed to process")
  print("...all files ready")
  print()

# Create the model
# See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
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
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
)

# # TODO Make these files available on the local file system
# # You may need to update the file paths
# files = [
#   upload_to_gemini("example.pdf", mime_type="application/pdf"),
# ]

# # Some files have a processing delay. Wait for them to be ready.
# wait_for_files_active(files)

doc = pymupdf.open("example.pdf") # open a document
text = ""
for page in doc: # iterate the document pages
  text += page.get_text() # get plain text encoded as UTF-8

chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [
        text,
      ],
    },
  ]
)

response = chat_session.send_message("What is this file")

print(response.text)