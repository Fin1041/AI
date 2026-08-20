import streamlit as st
from google import genai
from pypdf import PdfReader
import os
import glob

# 1. 페이지 및 타이틀 설정
st.set_page_config(page_title="주택관리공단 대구경북지사", page_icon="📚")
st.title("📚 대구경북 기술업무 AI 챗봇")

# 2. Gemini API 키 확인
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("Gemini API 키가 설정되지 않았습니다. Streamlit Secrets 설정을 확인해주세요.")
    st.stop()

genai.configure(api_key=api_key)

# 3. 모든 PDF 파일에서 텍스트를 추출하는 함수 (캐싱 처리로 속도 최적화)
@st.cache_data
def load_all_pdfs_text():
    # 현재 폴더 및 subfolder 내의 모든 .pdf 파일 탐색
    pdf_files = glob.glob("*.pdf") + glob.glob("**/*.pdf", recursive=True)
    
    if not pdf_files:
        return None, []

    combined_text = ""
    file_names = []

    for file_path in pdf_files:
        filename = os.path.basename(file_path)
        file_names.append(filename)
        try:
            reader = PdfReader(file_path)
            combined_text += f"\n\n--- [문서 파일명: {filename}] ---\n"
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text:
                    combined_text += f"\n[페이지 {page_num}]\n" + text
        except Exception as e:
            st.error(f"'{filename}' 읽기 오류: {e}")

    return combined_text, file_names

# PDF 읽어오기
document_text, loaded_files = load_all_pdfs_text()

# 사이드바에 현재 검색 대상이 되는 PDF 파일 목록 표시 (사용자에게 원본 다운로드는 안 됨)
with st.sidebar:
    st.header("📄 학습된 규정집 목록")
    if loaded_files:
        st.success(f"총 {len(loaded_files)}개의 PDF 문서를 참조 중입니다.")
        for name in loaded_files:
            st.write(f"• {name}")
    else:
        st.warning("업로드된 PDF 파일이 없습니다.")

if not document_text:
    st.warning("GitHub 저장소에 PDF 파일(.pdf)을 하나 이상 업로드해 주세요.")
    st.stop()

# 4. 챗봇 대화 기록 유지
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 내용 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. 사용자 질문 처리
if user_input := st.chat_input("규정이나 법률에 대해 질문하세요 (예: 연차휴가 규정 알려줘):"):
    # 사용자 질문 화면 표시
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Gemini에 전달할 통합 프롬프트 (출처 파일명도 함께 언급하도록 지시)
    prompt = f"""
    너는 사내 통합 규정집 안내 도우미야.
    아래 제공된 [통합 규정 문서]만을 바탕으로 질문에 정확하고 친절하게 답변해줘.

    [답변 지침]
    1. 답변할 때 답변 내용의 출처가 되는 [문서 파일명]을 반드시 밝혀줘.
    2. 문서 내용에 없는 질문이라면 "해당 내용은 업로드된 규정집에 명시되어 있지 않습니다."라고 답변해줘.

    [통합 규정 문서]
    {document_text}

    [사용자 질문]
    {user_input}
    """

    with st.chat_message("assistant"):
        with st.spinner("여러 규정집을 검색 중입니다..."):
            model = genai.GenerativeModel('gemini-3.6-flash')
            response = model.generate_content(prompt)
            st.markdown(response.text)

    st.session_state.messages.append({"role": "assistant", "content": response.text})
