from git import Repo
import os
import shutil
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

from dotenv import load_dotenv
load_dotenv()

allowed_extensions = ['.py', '.ipynb', '.md']
model_name = "all-MiniLM-L6-v2"
model_kwargs={'device': 'cpu'}
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

def cloning(url):
    current_path = os.getcwd()
    last_name = url.split('/')[-1]
    clone_path = last_name.split('.')[0]
    repo_path = os.path.join(current_path,clone_path)
    chroma_path = f'{clone_path}-chroma'

    if not os.path.exists(repo_path):
        Repo.clone_from(url, to_path=repo_path)
    return repo_path,chroma_path

def extract_all_files(repo_path):
        root_dir = repo_path
        docs = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for file in filenames:
                file_extension = os.path.splitext(file)[1]
                if file_extension in allowed_extensions:
                    try: 
                        loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                        docs.extend(loader.load_and_split())
                    except Exception as e:
                        pass
        return docs

def chunk_files(docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        texts = text_splitter.split_documents(docs)
        num_texts = len(texts)
        return texts

def create_embeddings(texts):
    embeddings = HuggingFaceEmbeddings(model_name= model_name,model_kwargs=model_kwargs)
    return embeddings


def load_db(texts, embeddings,repo_path,chroma_path):
    if os.path.exists(chroma_path):
         vectordb = Chroma(embedding_function=embeddings, persist_directory=chroma_path)
    else:
        vectordb = Chroma.from_documents(texts, embedding=embeddings, persist_directory=chroma_path)
        vectordb.persist()
    return vectordb

def retrieve_results(query,vectordb):
        memory = ConversationSummaryMemory(llm=llm, memory_key = "chat_history", return_messages=True)
        qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k":8}), memory=memory)
        result = qa(query)
        return result['answer']


