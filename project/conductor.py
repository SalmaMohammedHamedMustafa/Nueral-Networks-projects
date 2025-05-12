import os
import json
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import webrtcvad
import threading
import queue
from google.cloud import texttospeech
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
from langchain_openai import ChatOpenAI

# Environment setup
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/salma/college/nueral_networks/Dr_mohsen/project/interviewai-459420-ddd596af82a8.json"  # Update with your path
OPENAI_API_KEY = ""  # Replace with your OpenAI API key

# Initialize clients
tts_client = texttospeech.TextToSpeechClient()
stt_client = OpenAI(api_key=OPENAI_API_KEY)

# Output directory
output_dir = "./ai-agent-output"
os.makedirs(output_dir, exist_ok=True)

# Pydantic models
class InterviewInteraction(BaseModel):
    question: str = Field(..., title="The interview question in Egyptian Arabic")
    audio_file: str = Field(..., title="Path to the generated question audio file")
    response_audio_file: str = Field(..., title="Path to the recorded response audio file")
    transcribed_response: str = Field(..., title="Transcribed response")

class InterviewSession(BaseModel):
    interactions: List[InterviewInteraction] = Field(..., title="List of question-response interactions", min_items=1)

class Evaluation(BaseModel):
    score: int = Field(..., ge=1, le=5, title="Score from 1 to 5")
    feedback: str = Field(..., title="Brief feedback on the response")

class InterviewReport(BaseModel):
    evaluations: List[Evaluation] = Field(..., title="List of evaluations for each question")

    

def record_audio_vad(filename: str, sample_rate: int = 16000, frame_duration: int = 30, 
                     silence_duration: float = 1, max_duration: float = 60, 
                     initial_thinking_time: float = 3) -> str:
    """
    Record audio using Voice Activity Detection (VAD) with improved stopping mechanism and initial thinking time.
    
    Args:
        filename (str): Path to save the output audio file
        sample_rate (int): Audio sample rate (default: 16000)
        frame_duration (int): Duration of each audio frame in milliseconds (default: 30)
        silence_duration (float): Silence duration to stop recording (default: 1 second)
        max_duration (float): Maximum recording duration (default: 60 seconds)
        initial_thinking_time (float): Initial delay before starting silence detection (default: 3 seconds)
    
    Returns:
        str: Path to the saved audio file
    """
    # Setup VAD
    vad = webrtcvad.Vad()
    vad.set_mode(3)  # Most aggressive VAD mode

    # Calculation of frame and chunk sizes
    chunk_size = int(sample_rate * frame_duration / 1000)  # Samples per frame
    silence_frames = int(silence_duration * 1000 / frame_duration)  # Silent frames to stop
    max_frames = int(max_duration * 1000 / frame_duration)  # Max frames to record
    initial_thinking_frames = int(initial_thinking_time * 1000 / frame_duration)  # Initial thinking frames

    # Queues and flags for thread-safe communication
    audio_queue = queue.Queue()
    stop_event = threading.Event()
    
    def audio_callback(indata, frames, time, status):
        """Callback for audio input stream"""
        if status:
            print(status)
        
        # Convert to bytes for VAD processing
        frame = indata[:, 0].tobytes()
        
        # Check for speech activity
        try:
            is_speech = vad.is_speech(frame, sample_rate)
            audio_queue.put((indata.copy(), is_speech))
        except Exception as e:
            print(f"VAD processing error: {e}")
    
    def recording_manager():
        """Manage recording process in a separate thread"""
        audio_data = []
        silence_counter = 0
        frame_counter = 0
        
        while not stop_event.is_set():
            try:
                # Wait for audio data with a timeout
                data, is_speech = audio_queue.get(timeout=1)
                
                # Track speech and silence
                audio_data.append(data)
                frame_counter += 1
                
                # Skip silence detection during initial thinking time
                if frame_counter <= initial_thinking_frames:
                    continue
                
                if is_speech:
                    silence_counter = 0
                else:
                    silence_counter += 1
                
                # Stop conditions
                if (silence_counter >= silence_frames) or (frame_counter >= max_frames):
                    stop_event.set()
            
            except queue.Empty:
                # If no audio received, check for stop conditions
                if frame_counter >= max_frames + initial_thinking_frames:
                    stop_event.set()
        
        # Combine and save audio
        if audio_data:
            recording = np.concatenate(audio_data, axis=0)
            wavfile.write(filename, sample_rate, recording)
            print(f"Recording saved to {filename}")
        else:
            print("No audio recorded.")
    
    # Start recording
    print(f"Recording... Preparing with {initial_thinking_time} seconds of thinking time.")
    print("Please begin speaking after the initial pause.")
    
    # Start audio input stream
    with sd.InputStream(callback=audio_callback, 
                        channels=1, 
                        samplerate=sample_rate, 
                        dtype='int16', 
                        blocksize=chunk_size):
        
        # Start recording manager thread
        manager_thread = threading.Thread(target=recording_manager)
        manager_thread.start()
        
        # Wait for recording to complete
        manager_thread.join(timeout=max_duration + initial_thinking_time + 2)
        
        # Ensure everything is stopped
        stop_event.set()
        manager_thread.join()
    
    print("Recording stopped.")
    return filename

# Regular functions for audio operations
def text_to_speech(text: str, output_file: str) -> str:
    """Convert text to speech using Google Cloud TTS and save the audio."""
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ar-XA",
        name="ar-XA-Chirp3-HD-Achernar"
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        effects_profile_id=["small-bluetooth-speaker-class-device"],
        pitch=0,
        speaking_rate=1
    )
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    with open(output_file, "wb") as out:
        out.write(response.audio_content)
    os.system(f"mpg123 {output_file} || afplay {output_file} || start {output_file}")
    return output_file

def speech_to_text(audio_file: str) -> str:
    """Transcribe audio using OpenAI Whisper."""
    try:
        with open(audio_file, "rb") as f:
            transcript = stt_client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        return transcript.text
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

# Tool definitions wrapping the regular functions
@tool
def text_to_speech_tool(text: str, output_file: str) -> str:
    """Convert text to speech using Google Cloud TTS and save the audio."""
    return text_to_speech(text, output_file)

@tool
def record_audio_tool(filename: str, sample_rate: int = 16000) -> str:
    """Record audio from the microphone using VAD and save as a WAV file."""
    return record_audio_vad(filename, sample_rate=sample_rate)

@tool
def speech_to_text_tool(audio_file: str) -> str:
    """Transcribe audio using OpenAI Whisper."""
    return speech_to_text(audio_file)

# Tool to conduct the interview using regular functions
@tool
def conduct_interview_tool(script_file: str) -> dict:
    """Conduct the interview by reading the script, recording responses with VAD, and transcribing them."""
    with open(script_file, 'r', encoding='utf-8') as f:
        script_data = json.load(f)
    questions = [line['line'] for line in script_data['script']]

    interactions = []
    for i, question in enumerate(questions, start=1):
        print(f"\nQuestion {i}: {question}")
        audio_file = os.path.join(output_dir, f"question_{i}.mp3")
        
        # Generate and play question audio
        text_to_speech(question, audio_file)
        print("Question audio played. Preparing to record response.")

        # Record response with VAD
        response_file = os.path.join(output_dir, f"response_{i}.wav")
        print("Recording started. Please speak now.")
        record_audio_vad(response_file)
        print("Recording stopped.")

        # Transcribe response
        transcription = speech_to_text(response_file)

        # Store interaction
        interactions.append({
            "question": question,
            "audio_file": audio_file,
            "response_audio_file": response_file,
            "transcribed_response": transcription
        })

    return {"interactions": interactions}

# Interview Conductor Agent
interview_conductor_agent = Agent(
    role="Interview Conductor",
    goal="\n".join([
        "To conduct a mock technical interview by speaking questions in Egyptian Arabic.",
        "To record and transcribe candidate responses using VAD.",
        "To produce a structured record of questions and answers."
    ]),
    backstory="\n".join([
        "The Interview Conductor is a key component of TechInterviewerAI, designed for Arabic-speaking job seekers in the MENA tech market.",
        "It delivers questions naturally and captures responses for evaluation using advanced VAD technology."
    ]),
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY),
    verbose=True,
    tools=[conduct_interview_tool, text_to_speech_tool, record_audio_tool, speech_to_text_tool]
)

# Evaluation Agent
evaluation_agent = Agent(
    role="Interview Evaluator",
    goal="\n".join([
        "To evaluate the candidate's responses based on correctness, clarity, and completeness.",
        "To generate a detailed report with scores and feedback."
    ]),
    backstory="\n".join([
        "The Interview Evaluator is an expert in assessing technical interview responses for Arabic-speaking candidates in the MENA tech market.",
        "It provides constructive feedback to help candidates improve."
    ]),
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY),
    verbose=True
)

# Evaluation Task
evaluation_task = Task(
    description="\n".join([
        "Read the interview session data from 'step_10_interview_session.json'.",
        "Evaluate each response for correctness, clarity, and completeness.",
        "Assign a score from 1 to 5 and provide brief feedback.",
        "Compile evaluations into a report."
    ]),
    expected_output="A JSON object with evaluations for each question, including scores and feedback.",
    output_json=InterviewReport,
    output_file=os.path.join(output_dir, "step_11_interview_report.json"),
    agent=evaluation_agent
)

# Interview Conductor Task
interview_conductor_task = Task(
    description="\n".join([
        "Conduct the interview by calling the conduct_interview_tool with the path to 'interviewer_script.json'.",
        "The tool will handle generating audio for each question, recording responses via VAD, transcribing them, and returning interaction data."
    ]),
    expected_output="A JSON object containing a list of interactions, each with the question, audio file paths, and transcribed response.",
    output_json=InterviewSession,
    output_file=os.path.join(output_dir, "step_10_interview_session.json"),
    agent=interview_conductor_agent
)

# Crew setup
crew = Crew(
    agents=[interview_conductor_agent, evaluation_agent],
    tasks=[interview_conductor_task, evaluation_task],
    process=Process.sequential
)

# Run the crew
if __name__ == "__main__":
    crew.kickoff()
    print("Interview conducted and evaluated. Check output files in", output_dir)