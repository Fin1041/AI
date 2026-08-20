import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import os

st.set_page_config(page_title="사내 규정/법률 챗봇", page_icon="🤖")
st.title("📖 사내 규정 질의응답 챗봇")

api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("Gemini API 키가 설정되지 않았습니다.")
    st.stop()

genai.configure(api_key=api_key)

@st.cache_data
def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

pdf_file = "rules.pdf"

if os.path.exists(pdf_file):
    document_text = load_pdf_text(pdf_file)
else:
    st.warning(f"'{pdf_file}' 파일이 없습니다. GitHub에 rules.pdf를 업로드해주세요.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("규정이나 법률에 대해 질문하세요:"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    prompt = f"""
    너는 사내 규정 안내 도우미야. 아래의 [문서 내용]만을 바탕으로 질문에 친절히 답변해줘.
    문서에 없는 내용이라면 문서에 명시되어 있지 않다고 답변해.

    [문서 내용]
    {document_text}

    [사용자 질문]
    {user_input}
    """

    with st.chat_message("assistant"):
        with st.spinner("규정집 확인 중..."):
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            st.markdown(response.text)
            
    st.session_state.messages.append({"role": "assistant", "content": response.text})
