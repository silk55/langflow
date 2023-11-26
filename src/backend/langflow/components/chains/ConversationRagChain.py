from langflow import CustomComponent
from langchain.chains import LLMChain
from typing import Optional, Union, Callable
from langflow.field_typing import (
    BasePromptTemplate,
    BaseLanguageModel,
    BaseMemory,
    Chain,
)
from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)
from langchain.schema.vectorstore import VectorStore
from langchain.schema import BasePromptTemplate, BaseRetriever, Document


class ConversationRagChainComponent(CustomComponent):
    display_name = "ConversationRagChain"
    description = "Chain to run queries against LLMs with rag"

    def build_config(self):
        optional_chain_type = [
            "stuff",
            "map_reduce",
            "map_rerank",
            "refine",
        ]
        return {
            "llm": {"display_name": "LLM"},
            "memory": {"display_name": "Memory"},
            "retriever": {"diplay_name": "Retriever"},
            "condense_question_prompt": {"display_name": "condense_question_prompt"},
            "qa_chain_prompt": {"display_name": "qa_chain_prompt"},
            "chain_type": {
                "display_name": "chain_type",
                "options": optional_chain_type,
                "value": optional_chain_type[0],
                },
            "return_resource_documents": {
                "display_name": "return_resource_documents",
            },
            "code": {"show": False},
        }

    def build(
        self,
        llm: BaseLanguageModel,
        memory: BaseMemory,
        retriever: BaseRetriever,
        chain_type: str,
        return_resource_documents: bool,
        condense_question_prompt: Optional[BasePromptTemplate] = None,
        qa_chain_prompt: Optional[BasePromptTemplate] = None,
    ) -> Union[Chain, Callable]:
        combine_docs_chain_kwargs={}
        if qa_chain_prompt:
            combine_docs_chain_kwargs={
                "prompt": qa_chain_prompt,
            }
        if condense_question_prompt:
            return ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                condense_question_llm=condense_question_prompt,
                chain_type=chain_type,
                return_source_documents=return_resource_documents,
                memory=memory,
                combine_docs_chain_kwargs=combine_docs_chain_kwargs,
            )
        return ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                chain_type=chain_type,
                return_source_documents=return_resource_documents,
                memory=memory,
                combine_docs_chain_kwargs=combine_docs_chain_kwargs,
            )
            
