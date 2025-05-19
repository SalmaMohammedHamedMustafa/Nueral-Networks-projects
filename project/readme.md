# InterviewAI System

## Project Overview

InterviewAI is an AI-powered technical interview system designed for Arabic-speaking job seekers in the MENA tech market. The system can process job descriptions, generate relevant technical interview questions in Egyptian Arabic, conduct the interview through speech synthesis and voice recognition, and evaluate responses.

## Components

The system consists of several interconnected components:

1. **Email Integration System** (`check_email.py`): Monitors an email inbox for incoming SSH key submissions, downloads attachments, and triggers subsequent processing.

2. **Job Analysis and Question Generation** (`interviewer.py`): Analyzes job descriptions to identify technical skills and requirements, searches for relevant interview questions, and generates a tailored interview script in Egyptian Arabic.

3. **Interview Conductor** (`conductor.py`): Uses the generated script to conduct the actual interview, converting text to speech in Arabic, recording and transcribing candidate responses, and evaluating performance.

## Features

- Email monitoring for automated interview setup
- Job description analysis for targeted questions
- Text-to-speech in Egyptian Arabic
- Voice Activity Detection (VAD) for natural conversation flow
- Speech-to-text transcription
- Automated response evaluation
- SSH key processing capability

## System Architecture

```
+----------------+     +----------------+     +----------------+
| Email Monitor  |     | Interviewer    |     | Conductor      |
| check_email.py |---->| interviewer.py |---->| conductor.py   |
+----------------+     +----------------+     +----------------+
      |                       |                      |
      v                       v                      v
  SSH Keys              Question Script        Interview Results
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- Google Cloud account with Text-to-Speech API enabled
- OpenAI API key
- Tavily API key
- Gmail account with App Password

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SalmaMohammedHamedMustafa/Nueral-Networks-projects.git
   cd interviewai
   ```

2. Install dependencies:
   ```bash
   pip install -r req.txt
   ```

3. Set up environment variables:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/credentials.json
   export OPENAI_API_KEY=your_openai_api_key
   export TAVILY_API_KEY=your_tavily_api_key
   export GOOGLE_API_KEY=your_google_api_key
   export GEMINI_API_KEY=your_gemini_api_key
   ```

### Configuration

1. Update email settings in `check_email.py`:
   - Replace `EMAIL` with your Gmail address
   - Replace `APP_PASSWORD` with your Gmail App Password
   - Update `USERNAME` with your system username

2. Update file paths in all scripts:
   - Replace credential paths
   - Update output directories

## Usage

### Starting the Email Monitor

```bash
python check_email.py
```

The system will monitor the specified email inbox for messages with the subject "SSH Key Submission from [name]" and process any JSON attachments.

### Running the Interview Process Directly

1. Generate interview questions:
   ```bash
   python interviewer.py --job-description "Your job description here"
   ```
   
   Alternatively, you can provide a JSON file:
   ```bash
   python interviewer.py --json-file path/to/info.json
   ```

2. Conduct the interview:
   ```bash
   python conductor.py
   ```

## Output Files

The system generates several output files during operation:

- `info.json`: Contains extracted job information
- `interviewer_script.json`: Generated interview questions
- `ai-agent-output/step_10_interview_session.json`: Record of the interview session
- `ai-agent-output/step_11_interview_report.json`: Evaluation report
- `email_downloader.log`: Log file for the email monitoring service

## Example Job Description JSON

```json
{
  "username": "candidate",
  "email": "candidate@example.com",
  "jobDescription": "Job title: LLM Engineer\n\nRequirements: Experience with Python, PyTorch, and Hugging Face. Knowledge of transformer architectures and fine-tuning methods. Experience with REST APIs and cloud deployment.",
  "sshKey": "ssh-rsa AAAA..."
}
```

## Security Notes

- The system processes SSH keys, so ensure proper security measures are in place
- Avoid committing API keys and credentials to version control
- The email password in `check_email.py` should be an App Password, not your main account password
