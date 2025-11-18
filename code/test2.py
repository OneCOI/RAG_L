from langchain_community.tools import TavilySearchResults
import os
import dotenv
from langchain_community.vectorstores import Chroma,FAISS
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader,WebBaseLoader
from langchain_ollama import OllamaEmbeddings

dotenv.load_dotenv()
os.environ['TAVILY_API_KEY']=os.getenv('TAVILY_API_KEY1')

#获取Tavliy搜索工具的实例
search=TavilySearchResults(max_results=3)

res=search.invoke('今天上海天气怎么样')
# print(res)


#步骤1：创建TextLoader的实例，并加载文档
loader=WebBaseLoader(
    'https://www.cuiliangblog.cn/detail/section/234349148'
)

docs=loader.load()

#步骤2：创建拆分器，并拆分
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

split_docs=text_splitter.split_documents(docs)

# print(split_docs)

#步骤3：创建嵌入模型
embedding_model=OllamaEmbeddings(
    model="bge-m3:latest"
)


#步骤4：将文档及嵌入模型传入到Chroma相关的结构中，进行数据的存储
db=FAISS.from_documents(
    documents=split_docs,
    embedding=embedding_model,
    # persist_directory='./FAISS-1'
)

#基于向量数据库获取检索器
retriever=db.as_retriever()


#进行数据的检索
docs2=retriever.invoke('低光图像增强的挑战')

# print(len(docs2))
# for doc in docs2:
#     print(doc)

# print(docs2[0].page_content)

from langchain_core.tools import create_retriever_tool

retriever_tool=create_retriever_tool(
    retriever=retriever,
    name='web_search',
    description='暗光增强论文'
)

tools=[search,retriever_tool]


from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

#获取大模型
model=ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen-plus"
)

#模型绑定工具
model_with_tools=model.bind_tools(tools)

#根据输入自动调用工具
messages=[HumanMessage(content='今天上海天气怎么样')]
response=model_with_tools.invoke(messages)
# print(response)
# print('-'*10)
# print(f'ContenString:{response.content}')
# print(f'ToolCalls:{response.tool_calls}')

from langchain_classic import hub

prompt = hub.pull("hwchase17/openai-functions-agent")

# print(prompt.messages)

from langchain_classic.agents import create_tool_calling_agent
from langchain_classic.agents import AgentExecutor

#创建Agent对象
agent=create_tool_calling_agent(model,tools,prompt)

#创建AgentExecutor对象
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=False)

from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.chat_history import BaseChatMessageHistory

from langchain_core.runnables.history import RunnableWithMessageHistory

#根据不同人的ID区分
store={}

#调取指定session_id对应的memory
def get_session_history(session_id:str)->BaseChatMessageHistory:

    if session_id not in store:
        store[session_id]=ChatMessageHistory()

    return store[session_id]


agent_with_chat_history=RunnableWithMessageHistory(
    runnable=agent_executor,
    get_session_history=get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
)

response=agent_with_chat_history.invoke(
    {'input':'Hi,我的名字是William'},
    config={'configurable':{'session_id':'123'}}
)

# print(response)

# response=agent_with_chat_history.invoke(
#     {'input':'我叫什么名字'},
#     config={'configurable':{'session_id':'123'}}
# )
while True:
    input1=input('请输入您的问题:')
    response=agent_with_chat_history.invoke(input={'input':input1},config={'configurable':{'session_id':'123'}})
    print('response:',response)



# print(response)