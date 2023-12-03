from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from OutputParser import person_parser, Person
from agents.twitter_lookup_agent import twitterlookup
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup
from langchain.llms.huggingface_hub import HuggingFaceHub

from third_parties.twitter import scrape_user_tweets


def ice_break(name: str) -> tuple[Person, str]:
    summary_template = """
       Given the LinkedIn information {linked_information} and Twitter information {twitter_information} about a person,I want you to create
       1.a short summary
       2.two interesting facts about them
       3. a topic that may interest them
       4. 2 creative ice breakers to start a conversation with them
       \n{format_instructions}
       """
    summary_prompt_template = PromptTemplate(
        input_variables=["linked_information", "twitter_information"],
        partial_variables={
            "format_instructions": person_parser.get_format_instructions()
        },
        template=summary_template,
    )
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")
    # llm=HuggingFaceHub(repo_id="flax-community/t5-base-cnn-dm")
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    linkedin_profile_url = lookup("gforguru")
    twitter_name = twitterlookup("gforguru1976 Bitcoin Cricket")
    linked_information = scrape_linkedin_profile(linkedin_profile_url)
    twitter_information = scrape_user_tweets(twitter_name)
    result = chain.run(
        linked_information=linked_information, twitter_information=twitter_information
    )
    return person_parser.parse(result), linked_information.get("profile_pic_url")


if __name__ == "__main__":
    result = ice_break("gforguru")
    print(result)
