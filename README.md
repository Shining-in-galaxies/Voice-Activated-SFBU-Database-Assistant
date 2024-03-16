# Voice-Activated SFBU Database Assistant

## Overview

This project combines voice recognition and text-to-speech technologies to create a voice-activated assistant similar to Siri or Alexa. It utilizes the SFBU database for retrieving information based on user queries. The assistant listens to the user's voice, processes the spoken words into text, retrieves relevant information from the SFBU database, and responds verbally with the information.

## Presentation

View presentation [PDF file](https://drive.google.com/file/d/1DAdmDQzpARJ6q8VqoFOUNYTGERM5WRLI/view?usp=sharing)

## Features

- Voice recognition via Whisper model.
- Information retrieval from the SFBU database using LangChain.
- Verbal responses using Google's Text-to-Speech API.
- Easy-to-use, voice-activated interface.

## Tech Stack

- **Python**: Primary programming language.
- **dotenv & os**: For managing environment variables.
- **speech_recognition & whisper**: For audio to text transcription.
- **torch & numpy**: For numerical operations and model inference.
- **pygame**: For audio playback.
- **google-cloud-texttospeech**: For text to speech conversion.
- **LangChain (langchain_openai, langchain_community)**: For building conversational AI systems, document processing, and text retrieval.
- **OpenAI API**: For accessing advanced language models (e.g., GPT-3.5).

## Environment

Developed and tested for Linux.

## Installation and Setup

1. Check Python Version
   Ensure you have Python 3 installed by running:
   `python3 --version`
   If Python is not installed, follow the instructions here: [Install Python on Ubuntu](https://www.makeuseof.com/install-python-ubuntu/).

2. Update your package list and install pip:
   `sudo apt update`
   `sudo apt install python3-pip`

3. Install virtual environment tools:
   `sudo apt install virtualenv virtualenvwrapper`

4. Configure the virtual environment:

   - Open file:
     `nano ~/.bashrc`
   - Add the following lines to the end of the file:
     `WORKON_HOME=$HOME/.virtualenvs`
     `VIRTUAL_ENVWRAPPER_PYTHON=/usr/bin/python3`
     `source /usr/share/virtualenvwrapper/virtualenvwrapper.sh`

5. Create a new virtual environment:
   `mkvirtualenv example`

6. Work on virtual environment:
   `workon example`

7. Clone this repository.

8. Navigate into the project directory:
   `cd ai-ecommerce-email-assistant`

9. Install the requirements:
   `pip install -r requirements.txt`

10. [OpenAI Migration](https://github.com/openai/openai-python/discussions/742):
    `openai migrate`

11. [Get your API key](https://beta.openai.com/account/api-keys)

12. Add OpenAI API Key to the Virtual Environment`s Environment Variables

    - Open or create an .env file within your virtual environment:
      `nano .env`
    - In the .env file, enter the following line, replacing your_api_key_here with your actual OpenAI API key:
      `OPENAI_API_KEY=your_api_key_here`
    - Activate the environment variables in your current session:
      `source .env`
    - Test if the OpenAI API Key was successfully added by printing it:
      `echo $OPENAI_API_KEY`
      If the command prints your API key, it has been successfully added to the environment variables.

13. Running the Application
    `python3 app.py`
    At this point, you can talk through microphone starting with `hey computer` to chat with the AI assistant.
