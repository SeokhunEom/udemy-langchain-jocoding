# from dotenv import load_dotenv
# from langchain.chat_models import ChatOpenAI
import streamlit as st
from langchain.llms import CTransformers

# load_dotenv()
# chat = ChatOpenAI()

llm = CTransformers(
    model="llama-2-7b-chat.ggmlv3.q2_K.bin",
    model_type="llama",
)

st.title('인공지능 시인')
content = st.text_input('시의 주제를 제시해주세요.')

if st.button('시 작성 요청하기'):
    with st.spinner('시 작성 중...'):
        # result = chat.predict(content + "에 대한 시를 써줘.")
        result = llm.predict("write a poem about" + content + ":")
        st.write(result)
