# Import necessary libraries
import streamlit as st
import openai
import os
from io import StringIO
import pandas as pd

from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.chains.router import MultiPromptChain
from langchain.llms import OpenAI
from langchain import PromptTemplate
import langchain

# Make sure to add your OpenAI API key in the advanced settings of streamlit's deployment
open_AI_key = os.environ.get('OPENAI_API_KEY')
openai.api_key = open_AI_key

st.header("Welcome to FiniBot!")
level = st.radio("Select your level:", ["Novice", "Expert"])


spectra = st.file_uploader("upload file", type={"csv", "txt"})
if spectra is not None:
    input = pd.read_csv(spectra)
st.write(input)

### Here, with some adjustments, copy-paste the code you developed for Question 1 in Assignment 3
##########################################################################

Output_template = """
You'll receive financial data. Follow the format below for your report:

Report the following from the provided data:
- "savings" is "Total Savings".
- "credit card debt" is "Monthly Debt".
- "income" is "Monthly Income".

Describe the financial situation by calculating the debt ratio. A high debt ratio is above 0.3.

Provide recommendations based on the financial situation.

Data: {input}
Stick to this output format:

- Total savings: ${input}
- Monthly debt: ${input}
- Monthly income: ${input}

- Address the client directly when explaining the financial situation.
- Address the client directly when giving recommendations.
"""
investment_template = """
Act as an investment advisor. Begin by congratulating them on their financial accomplishments.
Advise them to invest their money and provide an investment portfolio based on their savings in 5 different stocks using the available information.
Give a detailed description of the investment, including potential gains and the necessary steps.

""" + Output_template


debt_template = """
Act as a debt advisor. Approach them with politeness and caution, avoiding any guilt-tripping.
Discuss their financial situation and propose a debt repayment plan. This plan allocates 10% of their income towards monthly debt payments, among other options.
Present the plan in bullet points and estimate the number of years required to clear the debt.

""" + Output_template


routes = [
    {
        "name": "investment",
        "description": "Give investment advice if the debt ratio is less than 0.3",
        "prompt_template": investment_template
    },
    {
        "name": "debt",
        "description": "Give debt advice if the debt ratio is more than 0.3",
        "prompt_template": debt_template
    }
]


from langchain.llms import OpenAI
llm = OpenAI(temperature = 0.4)


destination_chains = {}
for p_info in routes:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain
default_chain = ConversationChain(llm=llm, output_key="text")

financial_prompt = """
Provide the financial summary in a {level} tone.
If the debt ratio is lower than 0.3, then act as an investment advisor; otherwise, act as a debt advisor.
"""

input_variables = ["input"]
prompt = PromptTemplate(template=financial_prompt, input_variables=input_variables)

MULTI_PROMPT_ROUTER_TEMPLATE = """\
Given a raw text input to a language model select the model prompt best suited for \
the input. You will be given the names of the available prompts and a description of \
what the prompt is best suited for. You may also revise the original input if you \
think that revising it will ultimately lead to a better response from the language \
model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \\ name of the prompt to use or "DEFAULT"
    "next_inputs": string \\ a modified version of the original input. It is modified to contai only: the "savings" value, the "debt" value, the "income" value, and the "summary" provided above.
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt names specified below OR \
it can be "DEFAULT" if the input is not well suited for any of the candidate prompts.
REMEMBER: "next_inputs" is not the original input. It is modified to contain: the "savings" value, the "debt" value, the "income" value, and the "summary" provided above.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (must include ```json at the start of the response) >>
<< OUTPUT (must end with ```) >>
"""

prompt = financial_prompt + MULTI_PROMPT_ROUTER_TEMPLATE
destinations = [f"{route['name']}: {route['description']}" for route in routes]
destinations_str = "\n".join(destinations)
router_template = prompt.format(destinations=destinations_str, level=level)
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
print(router_prompt.template)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

from langchain.chains.router import MultiPromptChain
from langchain.chains import ConversationChain

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=ConversationChain(llm=llm, output_key="text"),
    verbose=False,
)
##########################################################################


# UX goes here. You will have to encorporate some variables from the code above and make some tweaks.
# Add a "Submit" button
if st.button("Submit"):
        # Run FiniBot only when the button is clicked
    result = chain.run(input)
     # Display FiniBot's analys is and recommendation
    st.markdown("### FiniBot's Analysis and Recommendation:")
    st.markdown(result)
