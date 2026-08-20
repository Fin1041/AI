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
        "Gemini API 키가 설정되지 않았습니다.\n\n"
        "Streamlit Cloud → Manage app → Settings → Secrets에서 "
        "GEMINI_API_KEY를 확인해주세요."
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
# 6. PDF가 없는 경우
# ==========================================
if not document_text:

    st.warning(
        "GitHub 저장소에 PDF 파일(.pdf)을 하나 이상 업로드해주세요."
    )

    st.stop()


# ==========================================
# 7. 첫 화면 안내 메시지
# ==========================================
if "messages" not in st.session_state:

    welcome_message = """
안녕하십니까.

저는 **대구경북지사 기술업무 담당 AI 챗봇**입니다.

아래 기술업무 관련 문서를 근거로 답변해 드립니다.

"""

    # PDF 파일명 자동 표시
if loaded_files:

    for name in loaded_files:

        welcome_message += f"📋 {name}\n\n"

    else:

        welcome_message += "• 등록된 기술업무 문서가 없습니다.\n"

    welcome_message += """

궁금하신 사항을 질문해 주시면
**문서에 근거하여 정확하게 답변**해 드리겠습니다.

※ 업로드된 문서에 명시되지 않은 내용은
임의로 답변하지 않습니다.
"""

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": welcome_message
        }
    ]


# ==========================================
# 8. 기존 대화 표시
# ==========================================
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==========================================
# 9. 사용자 질문 입력
# ==========================================
user_input = st.chat_input(
    "규정이나 기술업무에 대해 질문하세요."
)


# ==========================================
# 10. 사용자가 질문했을 때만 실행
# ==========================================
if user_input:

    # --------------------------------------
    # 사용자 질문 표시
    # --------------------------------------
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):

        st.markdown(user_input)


    # ======================================
    # 11. Gemini에 전달할 프롬프트
    # ======================================
    prompt = f"""
너는 주택관리공단 대구경북지사의
기술업무 담당 AI 챗봇이다.

사용자의 질문에 대해 아래 [통합 기술업무 문서]의
내용만을 근거로 답변해야 한다.

[매우 중요한 답변 원칙]

1. 반드시 제공된 PDF 문서의 내용을 근거로 답변한다.

2. 문서에 없는 내용을 추측하거나 만들어내지 않는다.

3. 문서에서 답변 근거를 찾을 수 없는 경우에는
다음 문장을 그대로 사용한다.

"해당 내용은 업로드된 기술업무 문서에 명시되어 있지 않습니다."

4. 답변의 근거가 되는 PDF 파일명을 반드시 표시한다.

5. 가능한 경우 페이지 번호도 표시한다.

6. 여러 문서가 관련된 경우 관련된 모든 문서의
파일명을 표시한다.

7. 인터넷 검색이나 일반적인 지식을 사용하지 않는다.

8. 답변은 이해하기 쉬운 한국어로 작성한다.

9. 규정이나 업무절차를 설명할 때는
가능하면 항목별로 정리해서 답변한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[통합 기술업무 문서]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{document_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[사용자 질문]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{user_input}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[답변]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


    # ======================================
    # 12. Gemini 답변
    # ======================================
    with st.chat_message("assistant"):

        with st.spinner("📚 관련 기술업무 문서를 확인하고 있습니다..."):

            answer = None

            # 최대 3회 재시도
            for attempt in range(3):

                try:

                    response = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=prompt
                    )

                    answer = response.text

                    if answer:
                        break

                except Exception as e:

                    error_text = str(e)

                    # 503 오류인 경우 재시도
                    if (
                        "503" in error_text
                        or "UNAVAILABLE" in error_text
                    ):

                        if attempt < 2:

                            time.sleep(2 ** attempt)

                            continue

                    # 기타 오류
                    answer = (
                        "⚠️ Gemini AI 오류가 발생했습니다.\n\n"
                        f"오류 내용: `{error_text}`"
                    )

                    break


            # 모든 재시도 실패
            if not answer:

                answer = (
                    "⚠️ 현재 Gemini AI 서버가 일시적으로 "
                    "응답하지 않습니다.\n\n"
                    "잠시 후 다시 질문해 주세요."
                )


            # 답변 표시
            st.markdown(answer)


    # ======================================
    # 13. 답변 저장
    # ======================================
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
