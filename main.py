import os
import warnings
from typing import Dict

from openfabric_pysdk.utility import SchemaUtil

from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import Ray, State
from openfabric_pysdk.loader import ConfigClass

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re

import pprint
import google.generativeai as palm
from langchain.llms  import GooglePalm
from langchain.memory import ConversationBufferWindowMemory
from langchain.utilities import WikipediaAPIWrapper


#declare the wiki tool to be used by the agent
wiki = WikipediaAPIWrapper(features="html.parser")
tools = [Tool(
    name = "wikipedia",
    func = wiki.run,
    #setting up the description for the tool so that bot can understand how to use it.
    description= "you can use it when you need to answer any question that you dont know the answer of."

)]

#api key to connect with Google PaLM
api_key='AIzaSyDuD8AcadnybYQ1WvRavSHPsif8-95dSuc'

#Model used for the chatbot
model = GooglePalm(google_api_key = api_key, temperature = 0.7)

#Setting up the memory type for our conversational bot.
# k is the window size of the previous conversation it holds.
memory=ConversationBufferWindowMemory(k=2)


# Set up the custom template. Giving the bot its identity and instruction on how to use the tools and how to answer the questions.
template = """Your name is Cognitive-assistant. You ara an AI assistant who is an expert in answeing science questions. You have access to the following tools:

{tools}

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to explain the asnwer to the question in detail and use examples if possible.

Previous conversation between you and human:
{history}

New question: {input}
{agent_scratchpad}"""

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

prompt_with_history = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps", "history"]
)



#setting up a custom output parser so that we can extract final response from the LLM's response.
class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()

#creating LLMs chain 
llm_chain = LLMChain(llm=model, prompt=prompt_with_history)


tool_names = [tool.name for tool in tools]

#seting up the agent with all the configs.
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
    max_iterations = 2
)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=False, memory=memory)



############################################################
# Callback function called on update config
############################################################
def config(configuration: Dict[str, ConfigClass], state: State):
    # TODO Add code here
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: Ray, state: State) -> SimpleText:
    #making a continous string out of the "request" variable recieved
    resulting_string = ' '.join(request.text)
    #passing the string to the agent
    # print(resulting_string)
    output = agent_executor.run(resulting_string)
    # print(output)
    #passing the output to the frontend
    return SchemaUtil.create(SimpleText(), dict(text=output))
