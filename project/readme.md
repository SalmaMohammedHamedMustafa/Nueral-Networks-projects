# InterviewAI System

## Project Overview

InterviewAI is an AI-powered technical interview platform for Arabic-speaking candidates in the MENA tech market. It processes job descriptions, generates Egyptian Arabic interview questions, conducts interviews with speech synthesis and recognition, and evaluates responses.

## Components

1. **Email Integration System** (`check_email.py`): Monitors emails for SSH key submissions, downloads attachments, and triggers processing.
2. **Job Analysis and Question Generation** (`interviewer.py`): Analyzes job details, searches for questions, and creates tailored interview scripts.
3. **Interview Conductor** (`conductor.py`): Executes interviews with Google TTS, WebRTC VAD recording, OpenAI Whisper transcription, and evaluation.

## Features

- Email-based interview setup
- Targeted question generation from job descriptions
- Egyptian Arabic text-to-speech
- Voice Activity Detection (VAD)
- Speech-to-text transcription
- Automated response evaluation
- SSH key processing

## System Architecture

```
+----------------+     +----------------+     +----------------+     +----------------+
| Web Interface  |     | System Detects |     | Script Generator|     | Conductor      |
| Interviewee    |---->| Periodic Check |---->| Generates Script|---->| Conducts Interview|
| Applies via Web|     | for Requests   |     | from Job Desc  |     | & Outputs Report |
+----------------+     +----------------+     +----------------+     +----------------+
                                                            |
                                                            v
                                                    +----------------+
                                                    | User Gets Temp |
                                                    | Machine Access |
                                                    +----------------+
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- Google Cloud Text-to-Speech API
- OpenAI API key
- Tavily API key
- Gmail with App Password

### Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/SalmaMohammedHamedMustafa/Nueral-Networks-projects.git
   cd interviewai
   ```

2. Install dependencies:
   ```bash
   pip install -r req.txt
   ```

3. Set environment variables:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
   export OPENAI_API_KEY=your_openai_key
   export TAVILY_API_KEY=your_tavily_key
   export GOOGLE_API_KEY=your_google_key
   export GEMINI_API_KEY=your_gemini_key
   ```

### Configuration

- Update `check_email.py` with your Gmail address, App Password, and system username.
- Adjust file paths in scripts for credentials and output directories.

## Usage

### Start Email Monitor
```bash
python check_email.py
```
Monitors for "SSH Key Submission from [name]" emails and processes JSON attachments.

### Run Interview Process
1. Generate script:
   ```bash
   python interviewer.py --job-description "Your job description here"
   ```
   Or with JSON:
   ```bash
   python interviewer.py --json-file path/to/info.json
   ```

2. Conduct interview:
   ```bash
   python conductor.py
   ```

## Output Files

- `info.json`: Job details
- `interviewer_script.json`: Interview questions
- `ai-agent-output/step_10_interview_session.json`: Session data
- `ai-agent-output/step_11_interview_report.json`: Evaluation report
- `email_downloader.log`: Email monitoring log


## Security Notes

- Secure SSH key handling is critical.
- Avoid committing API keys or credentials to version control.
- Use Gmail App Password in `check_email.py`, not your main password.
