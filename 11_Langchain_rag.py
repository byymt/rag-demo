from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from  dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

load_dotenv()

loader = PyPDFLoader("rag_test_doc.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)
chunks = splitter.split_documents(pages)

embeddings = DashScopeEmbeddings(model= "text-embedding-v3" )

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_langchain_db",
    collection_name="pdf_rag"
)

retriever = vectorstore.as_retriever(search_kwargs={"k":1})

llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="qwen3.7-plus",
    temperature=0.0          # RAG答题用0,基于资料回答要稳定,不要发散
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是严谨的问答助手。只能根据下面提供的参考资料回答问题;"
               "资料中没有答案就回答'根据现有资料无法回答',禁止编造。"),
    ("human", "参考资料:\n{context}\n\n问题:{input}")
])

chain = LLMChain(llm=llm, prompt=prompt)

# 主流程:先检索,再把检索结果拼进context,最后调用链
question = "云实验室的简介"
docs = retriever.invoke(question)
context = "\n\n".join(doc.page_content for doc in docs)   # 3片拼成一个字符串
answer = chain.invoke({"context": context, "input": question})
print(answer["text"])
# results = retriever.invoke("云实验室的开放时间")
# for doc in results:
#     print(doc.page_content)
#     print("===")
# if __name__ =="__main__":
    # print(len(chunks))
    # # print(pages[0].page_content[:200])
    # print(pages[0].metadata)
    # print(chunks[0].page_content)