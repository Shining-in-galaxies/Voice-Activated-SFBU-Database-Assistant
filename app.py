from dotenv import load_dotenv
import os
import speech_recognition as sr
import whisper
import torch
import numpy as np
import pygame
import time
from google.cloud import texttospeech
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

# Environment variables for OpenAI and other APIs
openai_api_key = os.environ.get("OPENAI_API_KEY")
langchain_api_key = os.environ.get("LANGCHAIN_API_KEY")
# Setup Google Text-to-Speech
google_client = texttospeech.TextToSpeechClient()

# Initialize LangChain components
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0301", temperature=0.1)
embeddings = OpenAIEmbeddings()
loader = PyPDFLoader("2023Catalog.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
documents = text_splitter.split_documents(docs)
vector = Chroma.from_documents(documents, embeddings)
retriever = vector.as_retriever()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

# Whisper model setup for voice recognition
audio_model = whisper.load_model("base")

def listen_and_transcribe():
    r = sr.Recognizer()
    r.energy_threshold = 300
    r.pause_threshold = 0.8
    r.dynamic_energy_threshold = False

    with sr.Microphone(sample_rate=16000) as source:
        print("Listening...")
        audio = r.listen(source)
        torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
        result = audio_model.transcribe(torch_audio, language='english')
        return result["text"]

def synthesize_speech(text):
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = google_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    with open("output.mp3", "wb") as out:
        out.write(response.audio_content)
    pygame.mixer.init()
    sound = pygame.mixer.Sound("output.mp3")
    sound.play()
    time.sleep(sound.get_length())

def main():
    while True:
        user_input = listen_and_transcribe()
        if user_input.lower() == "quit":
            break
        result = qa.invoke({"question": user_input})
        response_text = result['answer']
        print("Response:", response_text)
        synthesize_speech(response_text)

if __name__ == "__main__":
    main()
