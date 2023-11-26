from __future__ import annotations

from langflow import CustomComponent
from typing import Optional
from langflow.field_typing import (
    BaseLanguageModel,
)
from langchain.tools import Tool
from langchain.schema import BaseRetriever, BaseChatMessageHistory
from langflow.custom.agent.autogpt import AutoGPT


class AutoGPTAgent(CustomComponent):
    display_name: str = "AutoGPT Agent"
    description: str = "AutoGPT Agent"

    def build_config(self):
        return {
            "tools": {"is_list": True, "display_name": "Tools"},
            "retriever": {"display_name": "Memory"},
            "ai_name": {"display_name": "AI name"},
            "ai_role": {"display_name": "AI role"},
            "llm": {"display_name": "LLM"},
            "human_in_the_loop": {"display_name": "Human in the loop"},
            "output_parser": {"display_name": "Output parser"},
            "chat_history_memory": {"display_name": "Chat history memory"},
            "max_loop_count": {"max_loop_count": "Max loop count"},
            "code": {"show": False},
        }

    def build(
        self,
        llm: BaseLanguageModel,
        tools: Tool,
        ai_name: str,
        ai_role: str,
        retriever: BaseRetriever,
        human_in_the_loop: Optional[bool] = False,
        chat_history_memory: Optional[BaseChatMessageHistory] = None,
        max_loop_count: Optional[int] = None,
    ) -> AutoGPT:
        return AutoGPT.from_llm_and_tools(
            ai_name=ai_name,
            ai_role=ai_role,
            tools=tools,
            llm=llm,
            memory=retriever,
            human_in_the_loop=human_in_the_loop,
            chat_history_memory=chat_history_memory,
            max_loop_count=max_loop_count,
        )

