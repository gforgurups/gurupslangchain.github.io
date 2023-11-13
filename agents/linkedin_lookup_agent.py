from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.utilities.serpapi import SerpAPIWrapper

from tools.tools import get_profile_url


def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")

    template = """Given the full name {name_of_person} I want you to get a link to their
    LinkedIn profile page. Your answer should contain only a URL"""

    prompt_template = PromptTemplate(input_variables=['name_of_person'], template=template)
    tools_for_agent=[
        Tool(func=get_profile_url, name="Crawl Google for LinkedIn Profile page",
             description="Useful when you need to get LinkedIn Page URL")
    ]
    agent = initialize_agent(tools=tools_for_agent,llm=llm,
                             AgentType = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True)
    linkedin_url = agent.run(prompt_template.format_prompt(name_of_person=name))
    return linkedin_url