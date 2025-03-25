import os
import pickle
import time
# import tiktoken
import langchain
# from langchain.llms import OpenAI
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_core.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, TransformChain
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

os.environ['GOOGLE_API_KEY'] = 'AIzaSyCC7aBSHvSFSvnJbvChJPvMeQO31hwIN90'

llm = GoogleGenerativeAI(model="gemini-pro", google_api_key='AIzaSyCC7aBSHvSFSvnJbvChJPvMeQO31hwIN90')

file_path = "vector_index_hugMovie.pkl"

def get_vectorIndex():
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorIndex = pickle.load(f)
            return vectorIndex
        
    else:
        return None
            
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=get_vectorIndex().as_retriever())

def get_embeddings():
    return HuggingFaceEmbeddings(model_name ='sentence-transformers/all-MiniLM-L6-v2')


def create_vectordb():
    loader = UnstructuredURLLoader(urls =[
        "https://en.wikipedia.org/wiki/Interstellar_(film)",
        "https://www.imdb.com/title/tt0816692/plotsummary/"
    ])
    data = loader.load()


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap= 200
    )

    docs = text_splitter.split_documents(data)

    vectorstore = FAISS.from_documents(docs, embeddings = get_embeddings)

    # Storing vector index create in local 
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

def ask_query(query):
    
        if(query != ""):
            print("Processing ....")
            result = chain.invoke({ "question": query}, return_only_outputs=True)
            print("Final Answer :", result['answer'], "\nSources      :", result['sources'])
            return result
        else:
            print("No question entered")

if __name__ == '__main__':
    print("Welcome to LLM Helper")
    query = input("Enter your query : ")

    result = ask_query(query);
    print("Processing done >>>\n")
    final_answer = result.get('answer', "No answer generated.")
    sources = result.get('sources', "No sources available.")

    print("Final Answer :", final_answer, "\nSources      :", sources)
    # print("Sources      :", sources)