from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
import requests
import os
import re
import streamlit as st
from dotenv import load_dotenv


# def get_video_title(video_id):
#     url = yt_url
#     res = requests.get(url)
#     if res.status_code == 200:
#         return res.json().get("title")
#     return "Unknown Video"

st.set_page_config(page_title="YouTube Chatbot", layout="wide")
st.title("YouTube Transcript Chatbot")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def render_transcript(final_text):
    st.markdown(
        f"""
    <div style="
        height:400px;
        overflow-y:auto;
        background:#1a1d26;
        padding:10px;
        border-radius:10px;
    ">
        {final_text}
    </div>

""",
unsafe_allow_html=True
    )

def render_chat(history):
    for q, a in history:
        st.markdown(f"""
        <div style="text-align:right; margin:10px;">
            <span style="background:#2563eb; padding:10px; border-radius:10px; color:white;">
                {q}
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="text-align:left; margin:10px;">
            <span style="background:#2d3748; padding:10px; border-radius:10px; color:white;">
                {a}
            </span>
        </div>
        """, unsafe_allow_html=True)



def get_video_id(yt_url):
    pattern = r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, yt_url)
    return match.group(1) if match else None


def get_trans(video_id):
    url = "https://youtube-transcript3.p.rapidapi.com/api/transcript"
    headers = {
        'x-rapidapi-key': os.getenv('RAPID_API_KEY'),
        'x-rapidapi-host': "youtube-transcript3.p.rapidapi.com"
    }
    params = {"videoId": video_id}
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    transcript = data.get("transcript", [])
    return " ".join([item['text'] for item in transcript])


def build_chain(final_text):
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=[". ", "? "]
    )
    chunks = splitter.create_documents([final_text])

    model_em = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, model_em)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={"k": 4})

    llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)
    model = llm

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
        Context:
          {context}
          Question: {question}

        Final Answer:
        """
    )

    def txt(retrieved_docs):
        return " ".join(doc.page_content for doc in retrieved_docs)

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(txt),
        'question': RunnablePassthrough()
    })

    return parallel_chain | prompt | model | StrOutputParser()

yt_url = st.text_input("Enter your URL")
video_id = get_video_id(yt_url)

if yt_url and not video_id:
    st.warning("Could not get the video ID. Check your URL.")
    st.stop()

if not video_id:
    st.stop()



col1,col2 = st.columns([1,1.2])


with col1:
    if st.session_state.get("last_video") != video_id:       
        with st.spinner("Loading transcript and building index..."):
            final_text = get_trans(video_id)

            if not final_text:
                st.error("Transcript is empty")
                st.stop()

            st.session_state.final_text = final_text
            st.session_state.chain = build_chain(final_text) 
            st.session_state.last_video = video_id
    if video_id:
        st.video(f"https://www.youtube.com/watch?v={video_id}")
                 
    st.subheader("Transcript Preview")
    if "final_text" in st.session_state:
        render_transcript(st.session_state.final_text)
    
    

with col2:
    st.markdown("###  Chat")

    chat_box = st.container()

    with chat_box:
        for q, a in st.session_state.chat_history:
            st.chat_message("user").write(q)
            st.chat_message("assistant").write(a)

   
    query = st.chat_input("Ask something about the video:")

    if query:
        with st.spinner("Thinking..."):
            if "chain" not in st.session_state:
                st.warning("Load a transcript first")
                st.stop()
            response = st.session_state.chain.invoke(query)

        st.session_state.chat_history.append((query, response))
        st.rerun()

