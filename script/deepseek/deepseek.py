
import os
BASE_URL = "https://api.deepseek.com"
API_KEY = "sk-9fc40e8ded4a45f5b9fc61b3330074d3"

deepseek_chat_model = "deepseek-chat"

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm1 = ChatOpenAI(model=deepseek_chat_model, api_key=API_KEY, base_url=BASE_URL)

ans = llm1.invoke([HumanMessage("hello")])

print(ans.content)