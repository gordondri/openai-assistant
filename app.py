import os
from io import BytesIO
import base64
import httpx
from pathlib import Path
from typing import List

from openai import AsyncAssistantEventHandler, AsyncOpenAI, OpenAI

from literalai.helper import utc_now

import chainlit as cl
from chainlit.config import config
from chainlit.element import Element, ElementBased


async_openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
sync_openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

assistant = sync_openai_client.beta.assistants.retrieve(
    os.environ.get("OPENAI_ASSISTANT_ID")
)

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID")

if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
    raise ValueError("ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID must be set")

config.ui.name = assistant.name
config.ui.theme.light

class EventHandler(AsyncAssistantEventHandler):

    def __init__(self, assistant_name: str) -> None:
        super().__init__()
        self.current_message: cl.Message = None
        self.current_step: cl.Step = None
        self.current_tool_call = None
        self.assistant_name = assistant_name

    async def on_text_created(self, text) -> None:
        self.current_message = await cl.Message(author=self.assistant_name, content="").send()

    async def on_text_delta(self, delta, snapshot):
        await self.current_message.stream_token(delta.value)

    async def on_text_done(self, text):
        await self.current_message.update()

    async def on_tool_call_created(self, tool_call):
        self.current_tool_call = tool_call.id
        self.current_step = cl.Step(name=tool_call.type, type="tool")
        self.current_step.language = "python"
        self.current_step.created_at = utc_now()
        await self.current_step.send()

    async def on_tool_call_delta(self, delta, snapshot): 
        if snapshot.id != self.current_tool_call:
            self.current_tool_call = snapshot.id
            self.current_step = cl.Step(name=delta.type, type="tool")
            self.current_step.language = "python"
            self.current_step.start = utc_now()
            await self.current_step.send()  
                 
        if delta.type == "code_interpreter":
            if delta.code_interpreter.outputs:
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        error_step = cl.Step(
                            name=delta.type,
                            type="tool"
                        )
                        error_step.is_error = True
                        error_step.output = output.logs
                        error_step.language = "markdown"
                        error_step.start = self.current_step.start
                        error_step.end = utc_now()
                        await error_step.send()
            else:
                if delta.code_interpreter.input:
                    await self.current_step.stream_token(delta.code_interpreter.input)


    async def on_tool_call_done(self, tool_call):
        self.current_step.end = utc_now()
        await self.current_step.update()

    async def on_image_file_done(self, image_file):
        image_id = image_file.file_id
        response = await async_openai_client.files.with_raw_response.content(image_id)
        image_element = cl.Image(
            name=image_id,
            content=response.content,
            display="inline",
            size="large"
        )
        if not self.current_message.elements:
            self.current_message.elements = []
        self.current_message.elements.append(image_element)
        await self.current_message.update()

# Function to encode an image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

@cl.step(type="tool")
async def speech_to_text(audio_file):
    response = await async_openai_client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )

    return response.text

# @cl.step(type="tool")
# async def generate_text_answer(transcription, images):
#     if images:
#         # Only process the first 3 images
#         images = images[:3]

#         images_content = [
#             {
#                 "type": "image_url",
#                 "image_url": {
#                     "url": f"data:{image.mime};base64,{encode_image(image.path)}"
#                 },
#             }
#             for image in images
#         ]

#         model = "gpt-4-turbo"
#         messages = [
#             {
#                 "role": "user",
#                 "content": [{"type": "text", "text": transcription}, *images_content],
#             }
#         ]
#     else:
#         model = "gpt-4o"
#         messages = [{"role": "user", "content": transcription}]

#     response = await async_openai_client.chat.completions.create(
#         messages=messages, model=model, temperature=0.3
#     )

#     return response.choices[0].message.content

@cl.step(type="tool")
async def text_to_speech(text: str, mime_type: str):
    CHUNK_SIZE = 1024
    
    if mime_type is None:
        mime_type = "audio/mp3"

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
    "Accept": mime_type,
    "Content-Type": "application/json",
    "xi-api-key": ELEVENLABS_API_KEY
    }

    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    async with httpx.AsyncClient(timeout=250.0) as client:
        response = await client.post(url, json=data, headers=headers)
        response.raise_for_status()  # Ensure we notice bad responses

        buffer = BytesIO()
        buffer.name = f"output_audio.{mime_type.split('/')[1]}"

        async for chunk in response.aiter_bytes(chunk_size=CHUNK_SIZE):
            if chunk:
                buffer.write(chunk)
        
        buffer.seek(0)
        return buffer.name, buffer.read()

async def upload_files(files: List[Element]):
    file_ids = []
    for file in files:
        uploaded_file = await async_openai_client.files.create(
            file=Path(file.path), purpose="assistants"
        )
        file_ids.append(uploaded_file.id)
    return file_ids


async def process_files(files: List[Element]):
    # Upload files if any and get file_ids
    file_ids = []
    if len(files) > 0:
        file_ids = await upload_files(files)

    return [
        {
            "file_id": file_id,
            "tools": [{"type": "code_interpreter"}, {"type": "file_search"}],
        }
        for file_id in file_ids
    ]

@cl.on_chat_start
async def start_chat():
    # Create a Thread
    thread = await async_openai_client.beta.threads.create()
    # Store thread ID in user session for later use
    cl.user_session.set("thread_id", thread.id)
    
@cl.set_starters
async def set_starters():
    
    # await cl.Message(content="Hello, Arkansas' newest batch of Disability Examiners!").send()
    # intro = (
    #     "I am the AI Assistant to Alex Watkins, Assistant Program Director of Training and Medical Liaison!\n\n"
    #     "You can ask me questions to help you get through the Disability Examiner Basic Training Program. Examples of questions are included below:\n\n"
    # )
    # output_name, output_audio = await text_to_speech(f"{intro}", "audio/webm")
    # output_audio_el = cl.Audio(
    #     name=output_name,
    #     mime="audio/webm",
    #     auto_play=True,
    #     content=output_audio,
    # )
    # answer_message = await cl.Message(content="").send()
    
    # answer_message.elements = [output_audio_el]
    # await answer_message.update()
    
    return [
        cl.Starter(
            label="Title II vs Title XVI",
            message="What are the key differences between the Title II and Title XVI disability programs, especially regarding funding and work criteria?",
            icon="/public/write.svg",
        ),
        cl.Starter(
            label="Responsibilities of a Disability Examiner",
            message="What are the primary responsibilities of a Disability Examiner (DE) in the disability determination process?",
            icon="/public/write.svg",
        ),

        cl.Starter(
            label="Sequential evaluation process",
            message="Can you explain the sequential evaluation process for determining disability for adults?",
            icon="/public/write.svg",
        )
    ]
    
    
    
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("dds", "Advocate4SSA"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None

@cl.on_message
async def main(message: cl.Message, audio_mime_type: str = None):
    thread_id = cl.user_session.get("thread_id")

    attachments = await process_files(message.elements)

    # Add a Message to the Thread
    oai_message = await async_openai_client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message.content,
        attachments=attachments,
    )

    # Create and Stream a Run
    async with async_openai_client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant.id,
        event_handler=EventHandler(assistant_name=assistant.name),
    ) as stream:
        await stream.until_done()
        
    # remove all markdown characters from the message
    # import re
    # def clean_markdown(text):
    #     # Define the characters to remove
    #     markdown_chars = r'[*_`~#>+={}$begin:math:display$$end:math:display$\\|-]'
        
    #     # Use regex to replace the characters with an empty string
    #     cleaned_text = re.sub(markdown_chars, '', text)
        
    #     return cleaned_text
    
    def clean_content(text):
        return text.replace('*', '').replace('#', '')
    print(f"Before cleaning: {stream.current_message.content}")
    stream.current_message.content = clean_content(stream.current_message.content)
    print(f"After cleaning: {stream.current_message.content}")
    
    # Synthesize audio from the last message
    output_name, output_audio = await text_to_speech(stream.current_message.content, audio_mime_type)
    
    print(f"Using mime type {audio_mime_type} for output audio")
    output_audio_el = cl.Audio(
        name=output_name,
        auto_play=True,
        mime=audio_mime_type,
        content=output_audio,
    )
    
    answer_message = await cl.Message(content="").send()

    answer_message.elements = [output_audio_el]
    await answer_message.update()


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    if chunk.isStart:
        buffer = BytesIO()
        # This is required for whisper to recognize the file type
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        # Initialize the session for a new audio stream
        cl.user_session.set("audio_buffer", buffer)
        # cl.user_session.set("audio_mime_type", chunk.mimeType)
        cl.user_session.set("audio_mime_type", "audio/mp3")

    # Write the chunks to a buffer and transcribe the whole audio at the end
    cl.user_session.get("audio_buffer").write(chunk.data)


@cl.on_audio_end
async def on_audio_end(elements: list[ElementBased]):
    # Get the audio buffer from the session
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)  # Move the file pointer to the beginning
    audio_file = audio_buffer.read()
    audio_mime_type: str = cl.user_session.get("audio_mime_type")
    
    if audio_mime_type is None or audio_mime_type == "audio/webm":
        audio_mime_type = "audio/mp3"

    print(f"Using mime type {audio_mime_type} for input audio")
    input_audio_el = cl.Audio(
        mime=audio_mime_type, 
        content=audio_file, 
        name=audio_buffer.name
    )
    await cl.Message(
        author="You",
        type="user_message",
        content="",
        elements=[input_audio_el, *elements],
    ).send()

    whisper_input = (audio_buffer.name, audio_file, audio_mime_type)
    transcription = await speech_to_text(whisper_input)

    msg = cl.Message(author="You", content=transcription, elements=elements)

    await main(message=msg, audio_mime_type=audio_mime_type)
