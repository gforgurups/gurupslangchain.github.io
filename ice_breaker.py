from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup
from langchain.llms.huggingface_hub import HuggingFaceHub
if __name__ == "__main__":
    print("Hello World")

    summary_template="""
    Given the LinkedIn information {information} about a person, I want you to create
    1. a short summary
    2. two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")
    #llm = HuggingFaceHub(repo_id="flax-community/t5-base-cnn-dm")
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    linkedin_profile_url = lookup("gforguru")
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url)
    print(chain.run(information=linkedin_data))

