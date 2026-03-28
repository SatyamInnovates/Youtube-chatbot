from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough

import os
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
import streamlit as st

st.set_page_config(page_title="youtube chatbot",layout="centered")
st.title("YouTube Transcript Chatbot")
# loader f
try:
    loader = TextLoader("transcripts.txt")
    transcripts_text = loader.load()
    final_text = transcripts_text[0].page_content

except Exception as e:
    print("error : ",e)

# splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=[". ", "? "]
)

chunks = splitter.create_documents([final_text])

# vector storing

model  = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(chunks, model)

# retriver

retriever = vector_store.as_retriever(search_type='similarity',search_kwargs={"k" : 5})



# augmentation

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational",
    huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
)

model = ChatHuggingFace(llm=llm)

compressor = LLMChainExtractor.from_llm(model) 

comp_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

prompt = ChatPromptTemplate.from_template(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """

)


parser = StrOutputParser()
def txt(retrived_docs):
    context_text = " ".join(doc.page_content for doc in retrived_docs)
    return context_text

parallel_chain = RunnableParallel({
    'context' : comp_retriever | RunnableLambda(txt),
    'question' : RunnablePassthrough()
}
)


main_chain = parallel_chain | prompt | model | parser



query = st.text_input("Ask something about the video:")

if st.button("Run"):
    if query:
        with st.spinner("Thinking..."):
            main_response  = main_chain.invoke(query)
            st.write(main_response)
    else:
        st.warning("Please enter a question")