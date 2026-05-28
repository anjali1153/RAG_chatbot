from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load the PDF document
loader = PyPDFLoader("data/Machine_Learning.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} documents.")

# Step 2: Split the documents into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size = 512, chunk_overlap = 64)

chunks = splitter.split_documents(documents)

#Step 3: Load embeddings model
embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"})

#Step 4: Create vector store and add documents
vectorestore = FAISS.from_documents(chunks, embeddings)

vectorestore.save_local("vectorstore/")