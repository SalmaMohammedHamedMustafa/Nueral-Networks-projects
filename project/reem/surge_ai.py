import os
import json
import nest_asyncio
import asyncio
import smtplib
import tkinter as tk
from tkinter import ttk, scrolledtext
from email.mime.text import MIMEText
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from pydantic import BaseModel, Field
from typing import List, Optional
from tavily import TavilyClient
from crawl4ai import AsyncWebCrawler
import threading

# Apply nest_asyncio for async compatibility
nest_asyncio.apply()

# Set up environment variables
required_env_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY", "GMAIL_ADDRESS", "GMAIL_APP_PASSWORD"]
for var in required_env_vars:
    if not os.getenv(var):
        raise ValueError(f"Environment variable {var} is not set")

# Create output directory
output_dir = "./ai-agent-output"
os.makedirs(output_dir, exist_ok=True)

# Define Pydantic models
class JobAnalysis(BaseModel):
    job_technical_level: str = Field(..., title="Technical level of the job (entry, mid, senior)")
    key_skills: List[str] = Field(..., title="List of key technical skills required", min_items=1)
    include_ps: bool = Field(..., title="Whether to include problem-solving questions")
    domain_knowledge: List[str] = Field(..., title="List of domain knowledge areas", min_items=0)

class SearchQueries(BaseModel):
    search_queries: List[str] = Field(..., title="Search queries for interview questions", min_items=1)

class SingleSearchResult(BaseModel):
    title: str = Field(..., title="Title of the search result")
    url: str = Field(..., title="URL of the resource")
    content: str = Field(..., title="Snippet or summary of the resource content")
    score: float = Field(..., title="Relevance score of the result")
    search_query: str = Field(..., title="The query that generated this result")

class AllSearchResults(BaseModel):
    results: List[SingleSearchResult] = Field(..., title="List of search results")

class SkillDetail(BaseModel):
    name: str = Field(..., title="Name of the skill or technology")
    description: Optional[str] = Field(None, title="Brief description of the skill or technology")

class SingleScrapedPage(BaseModel):
    page_url: str = Field(..., title="The URL of the scraped webpage")
    skills: List[SkillDetail] = Field(default_factory=list, title="List of skills extracted")
    technologies: List[SkillDetail] = Field(default_factory=list, title="List of technologies extracted")
    question_examples: List[str] = Field(..., title="List of example interview questions", min_items=1)
    source_type: Optional[str] = Field(None, title="Type of source (e.g., job posting, interview guide)")

    @classmethod
    def check_minimum_data(cls, values):
        skills = values.get('skills', [])
        technologies = values.get('technologies', [])
        question_examples = values.get('question_examples', [])
        if not skills and not technologies and not question_examples:
            raise ValueError("At least one of skills, technologies, or question_examples must be non-empty")
        return values

class AllScrapedPages(BaseModel):
    pages: List[SingleScrapedPage] = Field(..., title="List of scraped webpages", min_items=1)

class InterviewQuestion(BaseModel):
    question: str = Field(..., title="Interview question in professional Egyptian Arabic")
    type: str = Field(..., title="Question category (technical, problem-solving, scenario-based)")
    difficulty: str = Field(..., title="Difficulty level (easy, medium, hard)")

class InterviewScript(BaseModel):
    questions: List[InterviewQuestion] = Field(..., title="List of interview questions", min_items=5, max_items=10)

# Define tools
@tool
def search_engine_tool(query: str):
    """Search for resources related to technical skills and interview questions."""
    return search_client.search(query)

async def async_scrape_page(url: str) -> str:
    """Asynchronously scrape a webpage using crawl4ai."""
    try:
        print(f"Starting to scrape: {url}")
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url)
            if not result or not result.markdown:
                print("No content found when crawling the page")
                return "No content found on page"
            content = result.markdown
            print(f"Successfully crawled page with {len(content)} characters")
            return content
    except Exception as e:
        print(f"Error scraping page: {str(e)}")
        return f"Error scraping page: {str(e)}"

@tool
def web_scraping_tool_for_agent(page_url: str) -> str:
    """Synchronous wrapper for async scraping."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(async_scrape_page(page_url))

# Define email function
def send_email(recipient, interview_script):
    """Send an email with a dummy link and interview questions to the recipient."""
    try:
        dummy_meet_link = "https://example.com/mock-interview"

        body = f"""
        Dear User,

        Your interview questions have been generated successfully. Below is a summary:

        {json.dumps(interview_script, indent=2, ensure_ascii=False)}

        A mock interview session has been scheduled. Please join using the following link:
        {dummy_meet_link}

        Best regards,
        SURGE-AI Team
        """
        msg = MIMEText(body)
        msg['Subject'] = 'SURGE-AI: Your Interview Questions and Meeting Link'
        msg['From'] = os.environ["GMAIL_ADDRESS"]
        msg['To'] = recipient

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(os.environ["GMAIL_ADDRESS"], os.environ["GMAIL_APP_PASSWORD"])
            server.sendmail(msg['From'], msg['To'], msg.as_string())

        print(f"Email sent successfully to {recipient}")
    except Exception as e:
        print(f"Error sending email: {str(e)}")

# Define the GUI and workflow
class SurgeAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SURGE-AI: Generate Interview Questions")
        self.root.geometry("700x900")
        self.root.configure(bg="#0A0F0F")  # Dark sci-fi background

        # Initialize Tavily client
        global search_client
        search_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

        # Custom style for futuristic look
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", 
                        font=("Orbitron", 12), 
                        background="#00D4FF", 
                        foreground="#0A0F0F", 
                        padding=10, 
                        borderwidth=2, 
                        relief="flat")
        style.map("TButton", 
                  background=[('active', '#FFFFFF')], 
                  foreground=[('active', '#0A0F0F')])
        style.configure("TEntry", 
                        fieldbackground="#1C2526", 
                        foreground="#FFFFFF", 
                        font=("Helvetica", 12), 
                        padding=5)
        style.configure("TLabel", 
                        background="#0A0F0F", 
                        foreground="#00D4FF", 
                        font=("Orbitron", 12))

        # Main frame with padding for glow effect
        main_frame = tk.Frame(root, bg="#0A0F0F", bd=2, relief="flat")
        main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Title
        tk.Label(main_frame, 
                 text="SURGE-AI", 
                 bg="#0A0F0F", 
                 fg="#FFFFFF", 
                 font=("Orbitron", 24, "bold")).pack(pady=10)

        # Input fields
        tk.Label(main_frame, text="Job Position:", font=("Orbitron", 12), bg="#0A0F0F", fg="#00D4FF").pack()
        self.job_position_entry = ttk.Entry(main_frame, width=50)
        self.job_position_entry.pack(pady=5)

        tk.Label(main_frame, text="Requirements:", font=("Orbitron", 12), bg="#0A0F0F", fg="#00D4FF").pack()
        self.requirements_text = scrolledtext.ScrolledText(main_frame, width=50, height=10, bg="#1C2526", fg="#FFFFFF", insertbackground="#00D4FF", font=("Helvetica", 12))
        self.requirements_text.pack(pady=5)

        tk.Label(main_frame, text="Company Name:", font=("Orbitron", 12), bg="#0A0F0F", fg="#00D4FF").pack()
        self.company_name_entry = ttk.Entry(main_frame, width=50)
        self.company_name_entry.pack(pady=5)

        tk.Label(main_frame, text="Score Threshold:", font=("Orbitron", 12), bg="#0A0F0F", fg="#00D4FF").pack()
        self.score_th_entry = ttk.Entry(main_frame, width=50)
        self.score_th_entry.insert(0, "0.7")
        self.score_th_entry.pack(pady=5)

        tk.Label(main_frame, text="User Email:", font=("Orbitron", 12), bg="#0A0F0F", fg="#00D4FF").pack()
        self.email_entry = ttk.Entry(main_frame, width=50)
        self.email_entry.pack(pady=5)

        # Submit button
        self.submit_button = ttk.Button(main_frame, text="Generate Interview Questions", command=self.on_submit)
        self.submit_button.pack(pady=20)

        # Status label with fade-in effect
        self.status_label = tk.Label(main_frame, 
                                    text="", 
                                    wraplength=600, 
                                    bg="#0A0F0F", 
                                    fg="#FFFFFF", 
                                    font=("Orbitron", 12))
        self.status_label.pack(pady=10)
        self.status_alpha = 0.0

    def fade_in_status(self, message):
        self.status_label.config(text=message)
        if self.status_alpha < 1.0:
            self.status_alpha += 0.1
            self.status_label.config(fg="#FFFFFF")
            self.root.after(50, lambda: self.fade_in_status(message))
        else:
            self.status_alpha = 0.0

    def run_workflow(self, inputs):
        # Initialize LLMs
        llm = LLM(model="gpt-4o-mini", temperature=0)
        llm_advanced = LLM(model="gpt-4o", temperature=0)

        # Define agents
        input_processor_agent = Agent(
            role="Input Processor Agent",
            goal="Analyze job details to identify technical level, key skills, problem-solving relevance, and domain knowledge.",
            backstory="Starting point for SURGE-AI, supporting Arabic-speaking job seekers in the MENA tech market.",
            llm=llm,
            verbose=True,
        )

        search_query_generator = Agent(
            role="Search Query Generator",
            goal="Generate tailored search queries for interview questions.",
            backstory="Specialized in creating effective search queries for job-specific interview questions.",
            llm=llm,
            verbose=True,
        )

        research_agent = Agent(
            role="Research Agent",
            goal="Retrieve relevant resources for technical skills and interview questions.",
            backstory="Gathers high-quality resources for Arabic-speaking job seekers in the MENA tech market.",
            llm=llm,
            verbose=True,
            tools=[search_engine_tool]
        )

        web_scraper_agent = Agent(
            role="Web Scraper Agent",
            goal="Scrape webpages to extract skills, technologies, and interview questions.",
            backstory="Collects and analyzes web data for SURGE-AI, supporting question generation.",
            llm=llm,
            verbose=True,
            tools=[web_scraping_tool_for_agent]
        )

        question_generator_agent = Agent(
            role="Enhanced Question Generator",
            goal="Craft interview questions rooted in job scenarios and MENA tech trends.",
            backstory="Powers SURGE-AI’s Q&A module, producing high-fidelity interview prompts.",
            llm=llm_advanced,
            verbose=True
        )

        # Define tasks
        input_processor_task = Task(
            description=f"Analyze the job position '{inputs['job_position']}', requirements '{inputs['requirements']}', and company '{inputs['company_name']}' to produce a structured job analysis.",
            expected_output="JSON with job technical level, key skills, problem-solving relevance, and domain knowledge.",
            output_file=os.path.join(output_dir, "step_1_job_analysis.json"),
            output_json=JobAnalysis,
            agent=input_processor_agent
        )

        search_query_generator_task = Task(
            description="Generate up to 10 search queries based on job analysis.",
            expected_output="JSON with an array of search queries.",
            output_file=os.path.join(output_dir, "step_2_search_queries.json"),
            output_json=SearchQueries,
            agent=search_query_generator
        )

        research_task = Task(
            description="Search for resources using queries from step_2, prioritizing MENA relevance and score > {score_th}.",
            expected_output="JSON with search results including title, URL, content, score, and query.",
            output_json=AllSearchResults,
            output_file=os.path.join(output_dir, "step_3_research_results.json"),
            agent=research_agent
        )

        web_scraper_task = Task(
            description="Scrape URLs from step_3 to extract skills, technologies, and questions.",
            expected_output="JSON with scraped data including URL, skills, technologies, questions, and source type.",
            output_json=AllScrapedPages,
            output_file=os.path.join(output_dir, "step_4_scraped_data.json"),
            agent=web_scraper_agent
        )

        question_generator_task = Task(
            description="Generate 5–10 interview questions in Egyptian Arabic based on step_4 data.",
            expected_output="JSON with 5–10 questions, each with question, type, and difficulty.",
            output_json=InterviewScript,
            output_file=os.path.join(output_dir, "step_5_interview_script.json"),
            agent=question_generator_agent
        )

        # Define Crew
        interview_crew = Crew(
            agents=[
                input_processor_agent,
                search_query_generator,
                research_agent,
                web_scraper_agent,
                question_generator_agent,
            ],
            tasks=[
                input_processor_task,
                search_query_generator_task,
                research_task,
                web_scraper_task,
                question_generator_task,
            ],
            process=Process.sequential
        )

        # Run workflow
        try:
            result = interview_crew.kickoff(inputs=inputs)
            final_script_path = os.path.join(output_dir, "step_5_interview_script.json")
            if os.path.exists(final_script_path):
                with open(final_script_path, 'r') as f:
                    interview_script = json.load(f)
                print("\nGenerated Interview Questions:")
                for idx, q in enumerate(interview_script.get("questions", []), 1):
                    print(f"{idx}. {q['question']} (Type: {q['type']}, Difficulty: {q['difficulty']})")
                send_email(inputs["email"], interview_script)
            else:
                print("Error: Final interview script not found.")
                self.root.after(0, self.fade_in_status, "Error: Failed to generate interview questions.")
        except Exception as e:
            print(f"Error running workflow: {str(e)}")
            self.root.after(0, self.fade_in_status, f"Error: {str(e)}")

    def on_submit(self):
        # Validate inputs
        job_position = self.job_position_entry.get().strip()
        requirements = self.requirements_text.get("1.0", tk.END).strip()
        company_name = self.company_name_entry.get().strip()
        score_th_text = self.score_th_entry.get().strip()
        email = self.email_entry.get().strip()

        if not job_position or not requirements or not email:
            self.fade_in_status("Error: Job Position, Requirements, and Email are required.")
            return

        try:
            score_th = float(score_th_text)
        except ValueError:
            self.fade_in_status("Error: Score Threshold must be a number.")
            return

        # Disable button and update status
        self.submit_button.state(['disabled'])
        self.fade_in_status("Your input is being processed. Expect an email with the interview link shortly.")

        # Run workflow in a separate thread
        inputs = {
            "job_position": job_position,
            "requirements": requirements,
            "company_name": company_name,
            "score_th": score_th,
            "email": email
        }
        threading.Thread(target=self.run_workflow, args=(inputs,), daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = SurgeAIApp(root)
    root.mainloop()