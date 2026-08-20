import streamlit as st
from google import genai
from pypdf import PdfReader
import os
import glob
import time

# ==========================================
# 1. 페이지 설정
# ==========================================
st.set_page_config(
    page_title="주택관리공단 대구경북지사",
    page_icon="📚"
)

st.title("📚 대구경북 기술업무 AI 챗봇")


# ==========================================
# 2. Gemini API 키 확인
# ==========================================
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error(
        "Gemini API 키가 설정되지 않았습니다. "
        "Streamlit Secrets의 GEMINI_API_KEY를 확인해주세요."
    )
    st.stop()


# ==========================================
# 3. Gemini 클라이언트 생성
# ==========================================
try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"Gemini 클라이언트 생성 오류: {e}")
    st.stop()


# ==========================================
# 4. PDF 읽기 함수
# ==========================================
@st.cache_data
def load_all_pdfs_text():

    pdf_files = glob.glob("*.pdf")
    pdf_files += glob.glob("**/*.pdf", recursive=True)

    # 중복 제거
    pdf_files = list(dict.fromkeys(pdf_files))

    if not pdf_files:
        return None, []

    combined_text = ""
    file_names = []

    for file_path in pdf_files:

        filename = os.path.basename(file_path)

        file_names.append(filename)

        try:

            reader = PdfReader(file_path)

            combined_text += (
                "\n\n"
                "========================================\n"
                f"[문서 파일명: {filename}]\n"
                "========================================\n"
            )

            for page_num, page in enumerate(reader.pages, start=1):

                text = page.extract_text()

                if text:

                    combined_text += (
                        f"\n[페이지 {page_num}]\n"
                        f"{text}\n"
                    )

        except Exception as e:

            st.error(
                f"'{filename}' 읽기 오류: {e}"
            )

    return combined_text, file_names


# ==========================================
# 5. PDF 불러오기
# ==========================================
document_text, loaded_files = load_all_pdfs_text()


# ==========================================
# 6. 사이드바
# ==========================================


# ==========================================
# 7. PDF가 없는 경우
# ==========================================
if not document_text:

    st.warning(
        "GitHub 저장소에 PDF 파일(.pdf)을 하나 이상 업로드해주세요."
    )

    st.stop()


# ==========================================
# 8. 대화 기록 초기화
# ==========================================
if "messages" not in st.session_state:

    # 처음 접속했을 때 보여줄 안내문
    welcome_message = f"""
안녕하십니까.  
저는 **대구경북지사 기술업무 담당 AI 챗봇**입니다.

📚 **다음 기술업무 관련 규정 및 자료를 바탕으로 답변해 드립니다.**

"""

    # PDF 파일명 자동으로 추가
    if loaded_files:

        for name in loaded_files:
            welcome_message += f"• {name}\n"

    welcome_message += """
궁금하신 기술업무에 대해 질문해 주시면
**업로드된 문서에 근거하여 답변**해 드리겠습니다.

※ 문서에 명시되지 않은 내용은 임의로 답변하지 않습니다.
"""

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": welcome_message
        }
    ]


# ==========================================
# 9. 기존 대화 표시
# ==========================================
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==========================================
# 10. 사용자 질문 입력
# ==========================================
user_input = st.chat_input(
    "기술업무 규정이나 지침에 대해 질문해주세요 (예: 방재업무 근무기준)"
)


# ==========================================
# 11. 질문 처리
# ==========================================
if user_input:

    # --------------------------------------
    # 사용자 질문 저장
    # --------------------------------------
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # --------------------------------------
    # 사용자 질문 화면 표시
    # --------------------------------------
    with st.chat_message("user"):

        st.markdown(user_input)


    # --------------------------------------
    # Gemini에게 보낼 프롬프트
    # --------------------------------------
    prompt = f"""
너는 주택관리공단 대구경북지사의
사내 규정 및 기술업무 안내 AI 도우미야.

아래에 제공된 [통합 규정 문서]의 내용만을 근거로
사용자의 질문에 정확하고 이해하기 쉽게 답변해줘.

[답변 원칙]

1. 반드시 제공된 규정 문서의 내용만 사용해.

2. 문서에 없는 내용을 임의로 추측하거나 만들어내지 마.

3. 질문에 대한 근거가 문서에 없으면 다음 문장으로 답변해.

"해당 내용은 업로드된 규정집에 명시되어 있지 않습니다."

4. 답변할 때 반드시 근거가 되는 PDF 파일명을 표시해.

5. 가능하면 페이지 번호도 함께 표시해.

6. 여러 PDF의 내용이 관련되어 있다면
각각의 문서명을 표시해.

7. 제공된 PDF에 없는 내용은
인터넷이나 일반적인 지식을 이용하여 답변하지 마.

8. 답변은 한국어로 해.

[통합 규정 문서]

{document_text}

[사용자 질문]

{user_input}

[답변]
"""


    # --------------------------------------
    # Gemini 답변
    # --------------------------------------
with st.chat_message("assistant"):

    with st.spinner("📚 질문관련 답변을 검색하고 있습니다..."):

        answer = None

        models = [
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite"
        ]

        for model_name in models:

            try:

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                if response.text:
                    answer = response.text
                    break

            except Exception as e:

                error_text = str(e)

                # 503이면 다음 모델로 자동 전환
                if "503" in error_text or "UNAVAILABLE" in error_text:
                    continue

                # 그 외 오류는 바로 표시
                answer = (
                    "⚠️ Gemini AI 오류가 발생했습니다.\n\n"
                    f"오류 내용: {error_text}"
                )
                break

        # 모든 모델이 실패한 경우
        if not answer:

            answer = (
                "⚠️ 현재 AI 서버가 일시적으로 혼잡합니다.\n\n"
                "잠시 후 다시 질문해 주세요."
            )

        st.markdown(answer)


# ==========================================
# 답변 저장
# ==========================================
st.session_state.messages.append(
    {
        "role": "assistant",
        "content": answer
    }
)
