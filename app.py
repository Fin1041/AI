import streamlit as st
from google import genai

import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer

import os
import time


# ==================================================
# 1. 페이지 설정
# ==================================================

st.set_page_config(
    page_title="주택관리공단 대구경북지사",
    page_icon="📚",
    layout="centered"
)


st.title(
    "📚 대구경북 기술업무 AI 챗봇"
)


# ==================================================
# 2. Gemini API 확인
# ==================================================

api_key = st.secrets.get(
    "GEMINI_API_KEY"
)


if not api_key:

    st.error(
        "Gemini API 키가 설정되지 않았습니다.\n\n"
        "Streamlit Cloud → Manage app → Settings → Secrets에서 "
        "GEMINI_API_KEY를 확인해주세요."
    )

    st.stop()


# ==================================================
# 3. Gemini 클라이언트
# ==================================================

try:

    client = genai.Client(
        api_key=api_key
    )

except Exception as e:

    st.error(
        f"Gemini 클라이언트 생성 오류: {e}"
    )

    st.stop()


# ==================================================
# 4. 임베딩 모델
#
# 질문을 벡터로 바꾸기 위해 사용
# ==================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "intfloat/multilingual-e5-small"
    )

    return model


with st.spinner(
    "🔎 검색 시스템을 준비하고 있습니다..."
):

    embedding_model = load_embedding_model()


# ==================================================
# 5. 벡터 DB 위치
# ==================================================

VECTOR_FOLDER = "vector_db"

INDEX_PATH = os.path.join(
    VECTOR_FOLDER,
    "index.faiss"
)

DOCUMENTS_PATH = os.path.join(
    VECTOR_FOLDER,
    "documents.pkl"
)


# ==================================================
# 6. 벡터 DB 확인
# ==================================================

if not os.path.exists(
    INDEX_PATH
):

    st.error(
        "❌ vector_db/index.faiss 파일이 없습니다."
    )

    st.info(
        "관리자 PC에서 build_vector_db.py를 실행한 후 "
        "생성된 vector_db 폴더를 GitHub에 업로드해주세요."
    )

    st.stop()


if not os.path.exists(
    DOCUMENTS_PATH
):

    st.error(
        "❌ vector_db/documents.pkl 파일이 없습니다."
    )

    st.stop()


# ==================================================
# 7. 벡터 DB 불러오기
#
# 앱 실행 중 반복 로딩하지 않도록 캐시
# ==================================================

@st.cache_resource
def load_vector_database():

    index = faiss.read_index(
        INDEX_PATH
    )


    with open(
        DOCUMENTS_PATH,
        "rb"
    ) as f:

        documents = pickle.load(
            f
        )


    return index, documents


with st.spinner(
    "📚 규정집 검색 DB를 불러오는 중..."
):

    vector_index, documents = (
        load_vector_database()
    )


# ==================================================
# 8. 벡터 검색 함수
# ==================================================

def search_documents(
    query,
    top_k=6
):

    # ----------------------------------------------
    # 사용자 질문을 벡터로 변환
    # ----------------------------------------------

    query_embedding = (
        embedding_model.encode(
            [
                "query: " + query
            ],
            normalize_embeddings=True
        )
    )


    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )


    # ----------------------------------------------
    # FAISS 검색
    # ----------------------------------------------

    scores, indices = (
        vector_index.search(
            query_embedding,
            top_k
        )
    )


    results = []


    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:

            continue


        result = documents[
            int(idx)
        ].copy()


        result["score"] = float(
            score
        )


        results.append(
            result
        )


    return results


# ==================================================
# 9. 첫 화면
# ==================================================

if "messages" not in st.session_state:

    # ----------------------------------------------
    # 등록된 문서 목록
    # ----------------------------------------------

    filenames = []

    for document in documents:

        filename = document[
            "filename"
        ]

        if filename not in filenames:

            filenames.append(
                filename
            )


    welcome_message = """
안녕하십니까.

저는 **대구경북지사 기술업무 담당 AI 챗봇**입니다.

등록된 기술업무 규정집을 검색하여
질문과 관련성이 높은 규정을 찾아 답변해 드립니다.

### 📚 등록된 규정집
"""


    for filename in filenames:

        welcome_message += (
            f"- 📋 {filename}\n"
        )


    welcome_message += """

궁금하신 사항을 질문해 주세요.

답변은 관련 규정의 **문서명과 페이지**를
함께 표시합니다.

※ 등록된 문서에서 근거를 찾을 수 없는 내용은
임의로 답변하지 않습니다.
"""


    st.session_state.messages = [
        {
            "role": "assistant",
            "content": welcome_message
        }
    ]


# ==================================================
# 10. 기존 대화 표시
# ==================================================

for message in (
    st.session_state.messages
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ==================================================
# 11. 사용자 질문
# ==================================================

user_input = st.chat_input(
    "규정이나 기술업무에 대해 질문하세요."
)


# ==================================================
# 12. 질문 처리
# ==================================================

if user_input:

    # ----------------------------------------------
    # 사용자 질문 표시
    # ----------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )


    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_input
        )


# ==========================================
# AI 답변
# ==========================================

with st.chat_message("assistant"):

    # ------------------------------------------
    # 진행상황 표시
    # ------------------------------------------

    status = st.status(
        "📚 관련 규정을 검색하고 있습니다...",
        expanded=True
    )

    try:

        # ======================================
        # ① 벡터 검색
        # ======================================

        search_results = search_documents(
            user_input,
            top_k=6
        )

        # ======================================
        # ② 검색 결과 확인
        # ======================================

        if not search_results:

            status.update(
                label="❌ 관련 규정을 찾지 못했습니다.",
                state="complete",
                expanded=False
            )

            answer = (
                "해당 내용은 업로드된 "
                "기술업무 문서에 명시되어 있지 않습니다."
            )

            st.markdown(answer)

        else:

            # 검색된 규정 개수 표시
            status.update(
                label=f"✅ 관련 규정 {len(search_results)}건을 찾았습니다.",
                state="running",
                expanded=True
            )

            # ==================================
            # ③ 검색 결과 정리
            # ==================================

            context_parts = []

            for i, result in enumerate(
                search_results,
                start=1
            ):

                context_parts.append(
                    f"""
[검색결과 {i}]

문서명:
{result["filename"]}

페이지:
{result["page"]}페이지

검색 관련도:
{result["score"]:.3f}

내용:
{result["text"]}
"""
                )

            search_context = "\n".join(
                context_parts
            )

            # ==================================
            # ④ Gemini 프롬프트
            # ==================================

            prompt = f"""
너는 주택관리공단 대구경북지사의
기술업무 담당 AI 챗봇이다.

사용자의 질문에 대해 아래 [검색된 규정 문서]
내용만을 근거로 답변해야 한다.

[답변 원칙]

1. 검색된 규정 문서 내용을 근거로 답변한다.

2. 검색된 문서에 없는 내용을
추측하거나 만들어내지 않는다.

3. 문서에서 답변 근거를 찾을 수 없는 경우
다음 문장을 사용한다.

"해당 내용은 업로드된 기술업무 문서에 명시되어 있지 않습니다."

4. 답변의 근거가 되는 문서명을 반드시 표시한다.

5. 가능한 경우 페이지 번호를 표시한다.

6. 여러 문서가 관련된 경우
관련 문서를 모두 표시한다.

7. 인터넷 검색을 사용하지 않는다.

8. 일반적인 지식보다
제공된 규정 문서를 우선한다.

9. 답변은 이해하기 쉬운 한국어로 작성한다.

10. 규정이나 업무절차는 가능한 경우
번호나 항목을 사용하여 정리한다.

11. 검색된 규정 내용끼리 서로 다른 경우
임의로 하나를 선택하지 말고
차이가 있음을 알려준다.

━━━━━━━━━━━━━━━━━━━━━━━━━━

[검색된 관련 규정]

{search_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━

[사용자 질문]

{user_input}

━━━━━━━━━━━━━━━━━━━━━━━━━━

[답변]

"""

            # ==================================
            # ⑤ Gemini 답변 작성 시작
            # ==================================

            status.update(
                label="🤖 관련 규정을 확인했습니다. 답변을 작성하고 있습니다...",
                state="running",
                expanded=True
            )

            answer = None

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

                    if (
                        "503" in error_text
                        or "UNAVAILABLE" in error_text
                    ):

                        if attempt < 2:

                            status.update(
                                label=f"🔄 AI 서버 재시도 중... ({attempt + 1}/3)",
                                state="running",
                                expanded=True
                            )

                            time.sleep(2 ** attempt)
                            continue

                    answer = (
                        "⚠️ Gemini AI 오류가 발생했습니다.\n\n"
                        f"오류 내용: `{error_text}`"
                    )

                    break

            # ==================================
            # ⑥ 답변 완료
            # ==================================

            if not answer:

                answer = (
                    "⚠️ 현재 Gemini AI 서버가 "
                    "일시적으로 응답하지 않습니다.\n\n"
                    "잠시 후 다시 질문해 주세요."
                )

            status.update(
                label="✅ 답변 작성이 완료되었습니다.",
                state="complete",
                expanded=False
            )

            # ==================================
            # ⑦ 답변 표시
            # ==================================

            st.markdown(answer)

            # ==================================
            # ⑧ 답변 근거
            # ==================================

            st.markdown("---")

            st.markdown("### 📚 답변 근거")

            shown_sources = set()

            for result in search_results:

                source_key = (
                    result["filename"],
                    result["page"]
                )

                if source_key in shown_sources:
                    continue

                shown_sources.add(source_key)

                st.markdown(
                    f"📋 **{result['filename']}** "
                    f"— {result['page']}페이지"
                )

    except Exception as e:

        status.update(
            label="❌ 검색 중 오류가 발생했습니다.",
            state="error",
            expanded=True
        )

        answer = (
            "⚠️ 오류가 발생했습니다.\n\n"
            f"`{str(e)}`"
        )

        st.error(answer)

    # ==================================================
    # 13. 답변 저장
    # ==================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
