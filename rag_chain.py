from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


#1. Helper Functions

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#2. Load the question into embedding model.
def build_chain():
    #Load embedding+vectorstore
    embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-small-en-v1.5")
    vectorstore = FAISS.load_local("vectorstore/", embeddings,
                  allow_dangerous_deserialization=True)                
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    #Load LLM
    llm = Ollama(model = 'tinyllama')
    
    #Prompt Template
    prompt = PromptTemplate(input = ["context","question"],
                            template = """You are a helpful assistant for
                            answering questions based on the following context.
                            If the answer is not contained within the context, say you "I don't know".
                            Do not use external knowledge or make up answers apart from given context.
                            
                            Context: {context}
                            
                            Question: {question}
                            """)
    
    #Make LCEL (Langchain Expression Language) chain
    chain = (
        {"context":retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain,retriever