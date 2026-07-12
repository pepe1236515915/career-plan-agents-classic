import os

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task


MODEL = os.getenv("MODEL", "gemini/gemini-3.1-flash-lite")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "ไม่พบ GOOGLE_API_KEY ใน environment กรุณาตั้งค่าในไฟล์ .env "
        "เช่น GOOGLE_API_KEY=your_api_key_here"
    )


def build_llm() -> LLM:
    return LLM(
        model=MODEL,
        api_key=GOOGLE_API_KEY,
        temperature=0.3,
    )


llm = build_llm()


@CrewBase
class CareerPlanAgents:
    """Career Plan Agents crew."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def career_profile_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["career_profile_analyst"],
            llm=llm,
            verbose=True,
        )

    @agent
    def qualification_gap_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["qualification_gap_analyst"],
            llm=llm,
            verbose=True,
        )

    @agent
    def development_plan_designer(self) -> Agent:
        return Agent(
            config=self.agents_config["development_plan_designer"],
            llm=llm,
            verbose=True,
        )

    @agent
    def self_assessment_coach(self) -> Agent:
        return Agent(
            config=self.agents_config["self_assessment_coach"],
            llm=llm,
            verbose=True,
        )

    @task
    def analyze_profile_task(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_profile_task"],
        )

    @task
    def analyze_gap_task(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_gap_task"],
        )

    @task
    def create_development_plan_task(self) -> Task:
        return Task(
            config=self.tasks_config["create_development_plan_task"],
        )

    @task
    def write_final_report_task(self) -> Task:
        return Task(
            config=self.tasks_config["write_final_report_task"],
            output_file="report.md",
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )