import json

from langchain.agents import AgentExecutor
from typing import Dict, Optional, Union
import chainlit as cl

from typing import Dict, Optional, Union

import aiohttp

from chainlit.telemetry import trace_event
from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)

async def load_flow(schema: Union[Dict, str], tweaks: Optional[Dict] = None):
    from langflow import load_flow_from_json

    trace_event("load_langflow")

    if type(schema) == str:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                schema,
            ) as r:
                if not r.ok:
                    reason = await r.text()
                    raise ValueError(f"Error: {reason}")
                schema = await r.json()

    flow = load_flow_from_json(flow=schema, tweaks=tweaks)

    return flow


with open("./new_rag.json", "r") as f:
    schema = json.load(f)


@cl.on_chat_start
async def start():
    flow = await load_flow(schema=schema)
    cl.user_session.set("flow", flow)


@cl.on_message
async def main(message: cl.Message):
    # Load the flow from the user session
    flow = cl.user_session.get("flow")  # type: AgentExecutor

    # Enable streaming
    flow.question_generator.llm.streaming = True

    res = await flow.arun(
        question=message.content, callbacks=[cl.LangchainCallbackHandler()]
    )

    # Send the response
    await cl.Message(content=res).send()