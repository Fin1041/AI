import streamlit as st
from google import genai
from pypdf import PdfReader
import os
import glob


# =========================================================
# 1. 페이지 설정
# =========================================================

st.set_page_config(
    page_title="주택관리공단 대구경북지사 챗봇",
    page_icon="📚"
)

st.title("📚 대구경북 기술업무      AI 챗봇")


# =========================================================
# 2. Gemini API 키 확인
# =========================================================

api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error(
        "Gemini API 키가 설정되지 않았습니다. "
        "Streamlit Secrets 설정을 확인해주세요."
    )
    st.stop()


# Gemini 클라이언트 생성
client = genai.Client(api_key=api_key)


# =========================================================
# 3. PDF 파일에서 텍스트 추출
# =========================================================

@st.cache_data
def load_all_pdfs_text():

    # 현재 폴더 및 하위 폴더의 모든 PDF 검색
    pdf_files = glob.glob("*.pdf") + glob.glob("**/*.pdf", recursive=True)

    # PDF가 없는 경우
    if not pdf_files:
        return None, []

    combined_text = ""
    file_names = []

    # PDF 하나씩 읽기
    for file_path in pdf_files:

        filename = os.path.basename(file_path)
        file_names.append(filename)

        try:

            reader = PdfReader(file_path)

            combined_text += (
                f"\n\n--- [문서 파일명: {filename}] ---\n"
            )

            for page_num, page in enumerate(
                reader.pages,
                start=1
            ):

                text = page.extract_text()

                if text:
                    combined_text += (
                        f"\n[페이지 {page_num}]\n"
                        + text
                    )

        except Exception as e:

            st.error(
                f"'{filename}' 읽기 오류: {e}"
            )

    return combined_text, file_names


# =========================================================
# 4. PDF 불러오기
# =========================================================

document_text, loaded_files = load_all_pdfs_text()


# =========================================================
# 5. 사이드바에 PDF 목록 표시
# =========================================================

with st.sidebar:

    st.header("📄 학습된 규정집 목록")

    if loaded_files:

        st.success(
            f"총 {len(loaded_files)}개의 PDF 문서를 참조 중입니다."
        )

        for name in loaded_files:
            st.write(f"• {name}")

    else:

        st.warning(
            "업로드된 PDF 파일이 없습니다."
        )


# =========================================================
# 6. PDF가 없으면 프로그램 중지
# =========================================================

if not document_text:

    st.warning(
        "GitHub 저장소에 PDF 파일(.pdf)을 하나 이상 업로드해 주세요."
    )

    st.stop()


# =========================================================
# 7. 대화 기록 초기화
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================================
# 8. 이전 대화 내용 출력
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# =========================================================
# 9. 사용자 질문 입력
# =========================================================

user_input = st.chat_input(
    "규정이나 지침에 대해 질문하세요 "
    "(예: 방재근무 기준을 알려줘):"
)


# =========================================================
# 10. 질문이 입력되었을 때
# =========================================================

if user_input:

    # -----------------------------------------------------
    # 사용자 질문 화면에 표시
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):

        st.markdown(user_input)


    # -----------------------------------------------------
    # Gemini에게 전달할 프롬프트
    # -----------------------------------------------------

    prompt = f"""
너는 주택관리공단 대구경북지사의
사내 통합 규정집 안내 AI 도우미야.

반드시 아래 [통합 규정 문서]의 내용만을 근거로
사용자의 질문에 정확하고 친절하게 답변해줘.

[답변 원칙]

1. 규정집에 실제로 있는 내용만 답변해줘.

2. 답변의 근거가 되는 문서의 파일명을 반드시 알려줘.

3. 가능하면 해당 내용이 있는 페이지 번호도 알려줘.

4. 여러 문서에 관련 내용이 있다면
   관련된 문서들을 모두 알려줘.

5. 규정집에 질문에 대한 내용이 없다면
   다음 문장으로 답변해줘.

   "해당 내용은 업로드된 규정집에 명시되어 있지 않습니다."

6. 규정집에 없는 내용을 추측해서 답변하지 마.

7. 법률이나 규정의 내용을 임의로 만들어내지 마.

8. 답변은 이해하기 쉽게 작성해줘.

9. 답변 마지막에는 반드시
   "출처"를 표시해줘.

[통합 규정 문서]

{document_text}


[사용자 질문]

{user_input}
"""


    # -----------------------------------------------------
    # Gemini 답변 생성
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "여러 규정집을 검색 중입니다..."
        ):

            try:

                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt
                )

                answer = response.text

                st.markdown(answer)


            except Exception as e:

                answer = (
                    "죄송합니다. Gemini AI와 연결하는 과정에서 "
                    "오류가 발생했습니다.\n\n"
                    f"오류 내용: `{str(e)}`"
                )

                st.error(answer)


    # -----------------------------------------------------
    # 답변을 대화 기록에 저장
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
