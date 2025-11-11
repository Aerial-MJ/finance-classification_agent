import os
from Agent.configs.parse import args

LLM_MODEL = args.llm_model
LLM_BASE_URL = args.llm_base_url
LLM_API_KEY = args.llm_api_key

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm1 = ChatOpenAI(model=LLM_MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

ans = llm1.invoke([HumanMessage("hello")])

print(ans.content)