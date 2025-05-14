import os
import json
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import tool
from tavily import TavilyClient
from pydantic import BaseModel, Field
from typing import List, Optional
import nest_asyncio
from crawl4ai import AsyncWebCrawler
import asyncio



# --- Output Directory ---
output_dir = "/home/salma/college/nueral_networks/Dr_mohsen/project/ai-agent-output"
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

# --- Async Compatibility ---
nest_asyncio.apply()  # Enable nested async loops for compatibility

# --- Initialize Clients and LLMs ---
search_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
llm = LLM(model="gpt-4o-mini", temperature=0)  # Basic LLM for most agents
llm_advanced = LLM(model="gpt-4o", temperature=0)  # Advanced LLM for question generation

# --- Data Models ---
class JobAnalysis(BaseModel):
    job_technical_level: str = Field(..., title="Technical level of the job (entry, mid, senior)")
    key_skills: List[str] = Field(..., title="List of key technical skills required for the job", min_items=1)
    include_ps: bool = Field(..., title="Whether to include problem-solving questions in the interview")
    domain_knowledge: List[str] = Field(..., title="List of domain knowledge areas required", min_items=0)

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
    skills: List[SkillDetail] = Field(default_factory=list, title="List of skills extracted from the page")
    technologies: List[SkillDetail] = Field(default_factory=list, title="List of technologies extracted from the page")
    question_examples: List[str] = Field(..., title="List of example interview questions extracted", min_items=1)
    source_type: Optional[str] = Field(None, title="Type of source (e.g., job posting, interview guide, tech blog)")

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
    type: str = Field(..., title="Question category")
    difficulty: str = Field(..., title="Difficulty level")

class InterviewScript(BaseModel):
    questions: List[InterviewQuestion] = Field(..., title="List of interview questions", min_items=5, max_items=10)

# --- Tools ---
@tool
def search_engine_tool(query: str):
    """Search for resources related to technical skills and interview questions."""
    try:
        return search_client.search(query)
    except Exception as e:
        return f"Search error: {str(e)}"

async def async_scrape_page(url: str) -> str:
    """Asynchronously scrape a webpage using crawl4ai."""
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url)
            return result.markdown if result and result.markdown else "No content found on page"
    except Exception as e:
        return f"Error scraping page: {str(e)}"

@tool
def web_scraping_tool_for_agent(page_url: str) -> str:
    """Synchronous wrapper for async web scraping."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(async_scrape_page(page_url))

# --- Agents and Tasks ---
# **Input Processor Agent**
input_processor_agent = Agent(
    role="Input Processor Agent",
    goal="Analyze job details to identify technical level, skills, and problem-solving relevance.",
    backstory="Designed for Arabic-speaking job seekers in the MENA tech market.",
    llm=llm,
    verbose=True,
)

input_processor_task = Task(
    description="Analyze job position, requirements, and company name to identify technical level, key skills, problem-solving relevance, and domain knowledge.",
    expected_output="JSON object with job analysis.",
    output_file=os.path.join(output_dir, "step_1_job_analysis.json"),
    output_json=JobAnalysis,
    agent=input_processor_agent
)

# **Search Query Generator Agent**
search_query_generator = Agent(
    role="Search Query Generator",
    goal="Generate tailored search queries for interview questions.",
    backstory="Specialized in creating effective search queries for job-specific interview questions.",
    llm=llm,
    verbose=True,
)

search_query_generator_task = Task(
    description="Generate search queries based on job analysis.",
    expected_output="JSON file with search queries.",
    output_file=os.path.join(output_dir, "step_2_search_queries.json"),
    output_json=SearchQueries,
    agent=search_query_generator
)

# **Research Agent**
research_agent = Agent(
    role="Research Agent",
    goal="Retrieve relevant resources based on search queries.",
    backstory="Gathers high-quality resources for MENA tech job seekers.",
    llm=llm,
    verbose=True,
    tools=[search_engine_tool]
)

research_task = Task(
    description="Search for resources using queries from step_2_search_queries.json.",
    expected_output="JSON object with search results.",
    output_json=AllSearchResults,
    output_file=os.path.join(output_dir, "step_3_research_results.json"),
    agent=research_agent
)

# **Web Scraper Agent**
web_scraper_agent = Agent(
    role="Web Scraper Agent",
    goal="Scrape webpages to extract skills, technologies, and interview questions.",
    backstory="Extracts actionable insights for question generation.",
    llm=llm,
    verbose=True,
    tools=[web_scraping_tool_for_agent]
)

web_scraper_task = Task(
    description="Scrape URLs from step_3_research_results.json.",
    expected_output="JSON object with scraped data.",
    output_json=AllScrapedPages,
    output_file=os.path.join(output_dir, "step_4_scraped_data.json"),
    agent=web_scraper_agent
)

# **Question Generator Agent**
question_generator_agent = Agent(
    role="Enhanced Question Generator",
    goal="Craft interview questions based on scraped data.",
    backstory="Translates job specs into high-fidelity interview prompts for MENA tech trends.",
    llm=llm_advanced,
    verbose=True
)

question_generator_task = Task(
    description="Generate 5-10 interview questions in Egyptian Arabic based on step_4_scraped_data.json.",
    expected_output="JSON object with interview questions.",
    output_json=InterviewScript,
    output_file=os.path.join(output_dir, "step_5_interview_script.json"),
    agent=question_generator_agent
)

# --- Interview Crew ---
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

# --- Format Output for conductor.py ---
def format_script_for_conductor(input_script_file, output_script_file):
    """Convert step_5_interview_script.json to interviewer_script.json format expected by conductor.py."""
    try:
        with open(input_script_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        # Transform to conductor.py's expected format: {"script": [{"line": question}, ...]}
        script = [{"line": q["question"]} for q in input_data["questions"]]
        output_data = {"script": script}
        
        with open(output_script_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Formatted script saved to {output_script_file}")
    except FileNotFoundError:
        print(f"Error: Input script file {input_script_file} not found.")
    except Exception as e:
        print(f"Error formatting script: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Example input data
    job_position = "R&D AI ML Developer Intern"
    requirements = """Design and develop scalable AI-driven applications using Python..."""  # Replace with full requirements
    company_name = "Siemens"
    score_th = 0.7

    print("Starting Interview-eight crew...")
    try:
        result = interview_crew.kickoff(inputs={
            "job_position": job_position,
            "requirements": requirements,
            "company_name": company_name,
            "score_th": score_th
        })
        print("Interviewer crew completed successfully.")
    except Exception as e:
        print(f"Error during crew execution: {e}")

    # Format the output script for conductor.py
    input_script_file = os.path.join(output_dir, "step_5_interview_script.json")
    output_script_file = "/home/salma/college/nueral_networks/Dr_mohsen/project/interviewer_script.json"
    format_script_for_conductor(input_script_file, output_script_file)