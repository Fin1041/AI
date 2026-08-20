import streamlit as st
from google import genai
from pypdf import PdfReader
import os
import glob
import time
import numpy as np


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
# 3. Gemini 클라이언트
# ==========================================
try:

    client = genai.Client(api_key=api_key)

except Exception as e:

    st.error(
        f"Gemini 클라이언트 생성 오류: {e}"
    )

    st.stop()


# ==========================================
# 4. PDF 파일 찾기
# ==========================================
def get_pdf_files():

    pdf_files = glob.glob("*.pdf")
    pdf_files += glob.glob("**/*.pdf", recursive=True)

    # 중복 제거
    pdf_files = list(dict.fromkeys(pdf_files))

    return pdf_files


# ==========================================
# 5. PDF 페이지별 내용 읽기
# ==========================================
@st.cache_data
def load_pdf_pages():

    pdf_files = get_pdf_files()

    documents = []

    for file_path in pdf_files:

        filename = os.path.basename(file_path)

        try:

            reader = PdfReader(file_path)

            for page_num, page in enumerate(
                reader.pages,
                start=1
            ):

                text = page.extract_text()

                if text and text.strip():

                    documents.append(
                        {
                            "filename": filename,
                            "page": page_num,
                            "text": text.strip()
                        }
                    )

        except Exception as e:

            st.error(
                f"'{filename}' 읽기 오류: {e}"
            )

    return documents


# PDF 읽기
documents = load_pdf_pages()


# ==========================================
# 6. PDF 파일 목록
# ==========================================
loaded_files = sorted(
    list(
        set(
            document["filename"]
            for document in documents
        )
    )
)


# ==========================================
# 7. PDF가 없는 경우
# ==========================================
if not documents:

    st.warning(
        "GitHub 저장소에 PDF 파일(.pdf)을 하나 이상 업로드해주세요."
    )

    st.stop()


# ==========================================
# 8. 텍스트 길이 제한 함수
# ==========================================
def limit_text(text, max_chars=6000):

    if len(text) <= max_chars:

        return text

    return text[:max_chars]


# ==========================================
# 9. PDF 페이지 임베딩 생성
# ==========================================
@st.cache_data
def create_pdf_embeddings(documents):

    embeddings = []

    progress = st.progress(
        0,
        text="📚 PDF 의미검색 데이터를 준비하고 있습니다..."
    )

    total = len(documents)

    for i, document in enumerate(documents):

        text = limit_text(
            document["text"],
            6000
        )

        # 파일명도 함께 넣어서 검색 정확도 향상
        embedding_text = (
            f"문서명: {document['filename']}\n"
            f"페이지: {document['page']}\n"
            f"{text}"
        )

        success = False

        # 임베딩 최대 3회 시도
        for attempt in range(3):

            try:

                result = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=embedding_text
                )

                vector = result.embeddings[0].values

                embeddings.append(vector)

                success = True

                break

            except Exception:

                if attempt < 2:

                    time.sleep(
                        2 ** attempt
                    )

        if not success:

            embeddings.append(
                np.zeros(3072)
            )

        progress.progress(
            (i + 1) / total,
            text=(
                f"📚 PDF 의미검색 데이터 준비 중 "
                f"({i + 1}/{total})"
            )
        )

    progress.empty()

    return np.array(
        embeddings,
        dtype=np.float32
    )


# ==========================================
# 10. PDF 의미 벡터 생성
# ==========================================
document_embeddings = create_pdf_embeddings(
    documents
)


# ==========================================
# 11. 벡터 정규화
# ==========================================
def normalize_vectors(vectors):

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1

    return vectors / norms


normalized_document_embeddings = normalize_vectors(
    document_embeddings
)


# ==========================================
# 12. 질문 의미검색 함수
# ==========================================
def search_relevant_documents(
    question,
    top_k=8
):

    # --------------------------------------
    # 질문을 임베딩
    # --------------------------------------
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question
    )

    question_vector = np.array(
        result.embeddings[0].values,
        dtype=np.float32
    )

    # --------------------------------------
    # 질문 벡터 정규화
    # --------------------------------------
    question_norm = np.linalg.norm(
        question_vector
    )

    if question_norm == 0:

        return []

    question_vector = (
        question_vector / question_norm
    )


    # --------------------------------------
    # 의미 유사도 계산
    # --------------------------------------
    scores = (
        normalized_document_embeddings
        @ question_vector
    )


    # --------------------------------------
    # 높은 점수부터 정렬
    # --------------------------------------
    ranked_indices = np.argsort(
        scores
    )[::-1]


    results = []

    for index in ranked_indices:

        score = float(
            scores[index]
        )

        # 너무 관련성이 낮은 내용 제외
        if score < 0.25:

            continue

        document = documents[index].copy()

        document["score"] = score

        results.append(
            document
        )

        if len(results) >= top_k:

            break


    return results


# ==========================================
# 13. 첫 화면 안내
# ==========================================
if "messages" not in st.session_state:

    welcome_message = """
안녕하십니까.

저는 **대구경북지사 기술업무 담당 AI 챗봇**입니다.

아래 기술업무 관련 문서를 근거로 답변해 드립니다.

"""

    # PDF 파일명 한 줄씩 표시
    if loaded_files:

        for name in loaded_files:

            welcome_message += (
                f"📋 {name}\n\n"
            )

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
# 14. 기존 대화 표시
# ==========================================
for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ==========================================
# 15. 사용자 질문
# ==========================================
user_input = st.chat_input(
    "규정이나 기술업무에 대해 질문하세요."
)


# ==========================================
# 16. 질문 처리
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
    # 17. 의미 기반 PDF 검색
    # ======================================
    with st.spinner(
        "🔎 질문의 의미를 분석하여 관련 규정을 검색하고 있습니다..."
    ):

        try:

            relevant_documents = (
                search_relevant_documents(
                    user_input,
                    top_k=8
                )
            )

        except Exception as e:

            relevant_documents = []

            search_error = str(e)


    # ======================================
    # 18. 검색 결과가 없는 경우
    # ======================================
    if not relevant_documents:

        if "search_error" in locals():

            answer = (
                "⚠️ PDF 의미검색 중 오류가 발생했습니다.\n\n"
                f"오류 내용: `{search_error}`"
            )

        else:

            answer = (
                "해당 내용은 업로드된 기술업무 문서에 "
                "명시되어 있지 않습니다."
            )

        with st.chat_message("assistant"):

            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        st.stop()


    # ======================================
    # 19. 검색된 문서 내용 만들기
    # ======================================
    relevant_text = ""

    for document in relevant_documents:

        relevant_text += (
            "\n\n"
            "========================================\n"
            f"[문서 파일명: {document['filename']}]\n"
            f"[페이지: {document['page']}]\n"
            f"[검색 관련도: "
            f"{document['score']:.3f}]\n"
            "========================================\n"
            f"{document['text']}\n"
        )


    # ======================================
    # 20. Gemini 프롬프트
    # ======================================
    prompt = f"""
너는 주택관리공단 대구경북지사의
기술업무 담당 AI 챗봇이다.

사용자의 질문에 대해 아래 [의미검색으로 찾은 관련 문서]만을
근거로 답변해야 한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[매우 중요한 답변 원칙]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 반드시 아래 제공된 PDF 문서 내용만 근거로 답변한다.

2. 문서에 없는 내용을 추측하거나 만들어내지 않는다.

3. 질문에 대한 정확한 근거를 찾을 수 없는 경우
다음 문장으로 답변한다.

"해당 내용은 업로드된 기술업무 문서에 명시되어 있지 않습니다."

4. 답변의 근거가 되는 PDF 파일명을 반드시 표시한다.

5. 가능한 경우 페이지 번호를 반드시 표시한다.

6. 여러 문서가 관련된 경우 각각의 문서명과
페이지 번호를 표시한다.

7. 인터넷 검색이나 일반적인 지식을 사용하지 않는다.

8. 답변은 이해하기 쉬운 한국어로 작성한다.

9. 규정이나 업무절차를 설명할 때는
가능하면 항목별로 정리한다.

10. 검색된 문서 내용으로 확실하게 판단할 수 없는 경우
억지로 답변하지 않는다.

11. 검색 관련도가 높다고 해서 반드시 정답이라는 의미는 아니다.
실제 문서 내용을 확인하고 질문에 대한 근거가 있는지 판단한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[의미검색으로 찾은 관련 PDF 내용]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{relevant_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[사용자 질문]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{user_input}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[답변]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


    # ======================================
    # 21. Gemini 답변
    # ======================================
    with st.chat_message("assistant"):

        with st.spinner(
            "🤖 관련 규정을 확인하여 답변을 작성하고 있습니다..."
        ):

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

                    # 503 오류 재시도
                    if (
                        "503" in error_text
                        or "UNAVAILABLE" in error_text
                    ):

                        if attempt < 2:

                            time.sleep(
                                2 ** attempt
                            )

                            continue

                    answer = (
                        "⚠️ Gemini AI 오류가 발생했습니다.\n\n"
                        f"오류 내용: `{error_text}`"
                    )

                    break


            # 모든 시도 실패
            if not answer:

                answer = (
                    "⚠️ 현재 Gemini AI 서버가 일시적으로 "
                    "응답하지 않습니다.\n\n"
                    "잠시 후 다시 질문해 주세요."
                )


            # 답변 표시
            st.markdown(answer)


    # ======================================
    # 22. 대화 기록 저장
    # ======================================
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
