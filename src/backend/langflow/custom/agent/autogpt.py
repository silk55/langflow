import json
from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ChatMessageHistory
from langchain.schema import (
    BaseChatMessageHistory,
    Document,
)
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.tools.base import BaseTool
from langchain.tools.human.tool import HumanInputRun

from langchain_experimental.autonomous_agents.autogpt.output_parser import (
    AutoGPTOutputParser,
    BaseAutoGPTOutputParser,
)
from langchain_experimental.autonomous_agents.autogpt.prompt import AutoGPTPrompt
from langchain_experimental.autonomous_agents.autogpt.prompt_generator import (
    FINISH_NAME,
)
from langchain_experimental.pydantic_v1 import ValidationError
from langchain.callbacks.manager import Callbacks
from typing import List, Optional,Any, Dict, List, Union


class AutoGPT:
    """Agent class for interacting with Auto-GPT."""

    def __init__(
        self,
        ai_name: str,
        memory: VectorStoreRetriever,
        chain: LLMChain,
        output_parser: BaseAutoGPTOutputParser,
        tools: List[BaseTool],
        feedback_tool: Optional[HumanInputRun] = None,
        chat_history_memory: Optional[BaseChatMessageHistory] = None,
        max_loop_count: Optional[int] = None,
    ):
        self.ai_name = ai_name
        self.memory = memory
        self.next_action_count = 0
        self.chain = chain
        self.output_parser = output_parser
        self.tools = tools
        self.feedback_tool = feedback_tool
        self.chat_history_memory = chat_history_memory or ChatMessageHistory()
        self.max_loop_count = max_loop_count

    @classmethod
    def from_llm_and_tools(
        cls,
        ai_name: str,
        ai_role: str,
        memory: VectorStoreRetriever,
        tools: List[BaseTool],
        llm: BaseChatModel,
        human_in_the_loop: bool = False,
        output_parser: Optional[BaseAutoGPTOutputParser] = None,
        chat_history_memory: Optional[BaseChatMessageHistory] = None,
        max_loop_count: Optional[int] = None,
    ):
        prompt = AutoGPTPrompt(
            ai_name=ai_name,
            ai_role=ai_role,
            tools=tools,
            input_variables=["memory", "messages", "goals", "user_input"],
            token_counter=llm.get_num_tokens,
        )
        print(prompt)
        human_feedback_tool = HumanInputRun() if human_in_the_loop else None
        chain = LLMChain(llm=llm, prompt=prompt)
        return cls(
            ai_name,
            memory,
            chain,
            output_parser or AutoGPTOutputParser(),
            tools,
            feedback_tool=human_feedback_tool,
            chat_history_memory=chat_history_memory,
            max_loop_count=max_loop_count,
        )
    
    @property
    def input_keys(self) -> List[str]:
        """Keys expected to be in the chain input."""
        return ["goal"]

    @property
    def output_keys(self) -> List[str]:
        """Keys expected to be in the chain output."""
        return ["response"]
    
    def __call__(
        self,
        inputs: Union[Dict[str, Any], Any],
        callbacks: Callbacks = None,
    ) -> Dict[str, Any]:
        goals = self.prep_inputs(inputs=inputs)
        return {
            "response": self.run(goals=goals,callbacks=callbacks),
        }
    
    def prep_inputs(self, inputs: Union[Dict[str, Any], Any]) -> List[str]:
        goal = inputs.get("goal")
        return goal.split(";")

    def run(self, goals: List[str], callbacks: Callbacks=None,) -> str:
        user_input = (
            "Determine which next command to use, "
            "and respond using the format specified above:"
        )

        # Interaction Loop
        loop_count = 0
        while True:
            # Discontinue if continuous limit is reached
            loop_count += 1

            # Send message to AI, get response
            assistant_reply = self.chain.run(
                goals=goals,
                messages=self.chat_history_memory.messages,
                memory=self.memory,
                user_input=user_input,
                callbacks=callbacks,
            )

            # Print Assistant thoughts
            print(assistant_reply)
            self.chat_history_memory.add_message(HumanMessage(content=user_input))
            self.chat_history_memory.add_message(AIMessage(content=assistant_reply))

            # Get command name and arguments
            action = self.output_parser.parse(assistant_reply)
            tools = {t.name: t for t in self.tools}
            if action.name == FINISH_NAME:
                return action.args["response"]
            if action.name in tools:
                tool = tools[action.name]
                try:
                    observation = tool.run(action.args)
                except ValidationError as e:
                    observation = (
                        f"Validation Error in args: {str(e)}, args: {action.args}"
                    )
                except Exception as e:
                    observation = (
                        f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"
                    )
                result = f"Command {tool.name} returned: {observation}"
            elif action.name == "ERROR":
                result = f"Error: {action.args}. "
            else:
                result = (
                    f"Unknown command '{action.name}'. "
                    f"Please refer to the 'COMMANDS' list for available "
                    f"commands and only respond in the specified JSON format."
                )

            memory_to_add = (
                f"Assistant Reply: {assistant_reply} " f"\nResult: {result} "
            )
            if self.feedback_tool is not None:
                feedback = f"\n{self.feedback_tool.run('Input: ')}"
                if feedback in {"q", "stop"}:
                    print("EXITING")
                    return "EXITING"
                memory_to_add += feedback

            self.memory.add_documents([Document(page_content=memory_to_add)])
            self.chat_history_memory.add_message(SystemMessage(content=result))
            if self.max_loop_count and loop_count > self.max_loop_count:
                return assistant_reply