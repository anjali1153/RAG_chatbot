import gradio as gr
from rag_chain import build_chain

chain, retriever = build_chain()

def chat(message, history):
    # Get source docs separately for citation
    source_docs = retriever.invoke(message)
    sources = set(doc.metadata.get("source", "") for doc in source_docs)

    answer = chain.invoke(message)

    source_text = "\n\n📄 Sources: " + ", ".join(sources) if sources else ""
    return answer + source_text

gr.ChatInterface(
    fn=chat,
    title="Local RAG Chatbot",
    description="Chat with your PDFs — fully local, no API keys needed.",
).launch()