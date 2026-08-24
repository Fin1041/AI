import streamlit as st
from google import genai

import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer

import os
import time


# =========================================================
# 1. 페이지 설정
# =========================================================

st.set_page_config(
    page_title="대구경북 기술업무 AI 챗봇",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# =========================================================
# 2. 화면 디자인
# =========================================================

st.markdown(
    """
    <style>

    /* 전체 배경 */
    .stApp {
        background-color: #f5f8fc;
    }


    /* 화면 폭 */
    .main .block-container {
        max-width: 850px;
        padding-top: 20px;
        padding-bottom: 100px;
    }


    /* 상단 파란색 영역 */
    .top-box {
        background: linear-gradient(
            135deg,
            #0756c9,
            #318fe9
        );

        border-radius: 0 0 24px 24px;

        padding: 20px 24px;

        margin-bottom: 20px;

        color: white;

        box-shadow:
            0 6px 20px rgba(0, 80, 170, 0.18);
    }


    /* 기관명 */
    .agency-name {
        font-size: 15px;
        font-weight: 800;
        line-height: 1.3;
        margin-bottom: 8px;
    }


    /* 챗봇 제목 */
    .chatbot-title {
        font-size: 28px;
        font-weight: 900;
        letter-spacing: -1px;
        margin-bottom: 0;
    }


    /* 안내 카드 */
    .info-box {
        background: white;

        border: 1px solid #e1e8f0;

        border-radius: 20px;

        padding: 20px;

        margin-bottom: 18px;

        box-shadow:
            0 5px 18px rgba(30,70,110,0.07);
    }


    .info-title {
        color: #17365f;

        font-size: 19px;

        font-weight: 800;

        margin-bottom: 12px;
    }


    .info-text {
        color: #4b5c70;

        font-size: 15px;

        line-height: 1.8;
    }


    /* 선택된 규정집 */
    .selected-box {
        background: linear-gradient(
            135deg,
            #eaf6ff,
            #f7fbff
        );

        border: 1px solid #c6e2f8;

        border-radius: 18px;

        padding: 16px 18px;

        margin-bottom: 15px;
    }


    .selected-title {
        font-size: 13px;

        color: #3775a8;

        font-weight: 700;

        margin-bottom: 6px;
    }


    .selected-name {
        font-size: 16px;

        color: #173f6d;

        font-weight: 800;

        word-break: keep-all;
    }


    /* 규정집 목록 제목 */
    .list-title {
        color: #17365f;

        font-size: 19px;

        font-weight: 800;

        margin-top: 10px;

        margin-bottom: 10px;
    }


    /* 버튼 */
    div.stButton > button {

        width: 100%;

        border-radius: 14px !important;

        border: 1px solid #c8def5 !important;

        background-color: white !important;

        color: #175ca8 !important;

        font-weight: 700 !important;

        min-height: 45px !important;

        text-align: left !important;

        padding-left: 16px !important;

        margin-bottom: 4px;

    }


    div.stButton > button:hover {

        background-color: #eef6ff !important;

        border-color: #2c79d7 !important;

        color: #0756c9 !important;

    }


    /* 사용자 질문 */
    .user-question {

        background-color: #1263d6;

        color: white;

        padding: 13px 17px;

        border-radius: 18px 18px 5px 18px;

        margin: 12px 0;

        font-size: 15px;

        line-height: 1.6;

    }


    /* 답변 카드 */
    .answer-box {

        background-color: white;

        border: 1px solid #e1e8f0;

        border-radius: 18px;

        padding: 18px;

        margin-top: 10px;

        margin-bottom: 15px;

        box-shadow:
            0 4px 15px rgba(30,70,110,0.06);

    }


    .answer-title {

        color: #174d8f;

        font-size: 16px;

        font-weight: 800;

        margin-bottom: 10px;

    }


    /* 답변 근거 */
    .source-box {

        background-color: #eefaf8;

        border: 1px solid #cdebe5;

        border-radius: 14px;

        padding: 12px 14px;

        margin-top: 8px;

        color: #245c57;

        font-size: 13px;

        line-height: 1.6;

    }


    /* Streamlit 하단 */
    footer {
        visibility: hidden;
    }


    /* 상단 툴바 */
    [data-testid="stToolbar"] {
        visibility: hidden;
        height: 0;
    }


    /* 모바일 */
    @media (max-width: 600px) {

        .main .block-container {

            padding-left: 14px;
            padding-right: 14px;
            padding-top: 8px;

        }


        .top-box {

            padding: 18px 20px;

            border-radius: 0 0 22px 22px;

        }


        .chatbot-title {

            font-size: 25px;

        }


        .info-box {

            padding: 18px;

        }


        .info-text {

            font-size: 14px;

        }


        .selected-name {

            font-size: 15px;

        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 3. 상단 화면
# =========================================================

st.markdown(
    """
    <div class="top-box">

        <div class="agency-name">
            📚 주택관리공단 · 대구경북지사
        </div>

        <div class="chatbot-title">
            기술업무 AI 챗봇
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 4. Gemini API
# =========================================================

api_key = st.secrets.get("GEMINI_API_KEY")


if not api_key:

    st.error(
        "Gemini API 키가 설정되지 않았습니다."
    )

    st.info(
        "Streamlit Cloud → Manage app → Settings → Secrets에서 "
        "GEMINI_API_KEY를 확인해주세요."
    )

    st.stop()


# =========================================================
# 5. Gemini 클라이언트
# =========================================================

try:

    client = genai.Client(
        api_key=api_key
    )

except Exception as e:

    st.error(
        f"Gemini 클라이언트 생성 오류: {e}"
    )

    st.stop()


# =========================================================
# 6. 임베딩 모델
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "intfloat/multilingual-e5-small"
    )


with st.spinner(
    "🔎 검색 시스템을 준비하고 있습니다..."
):

    embedding_model = load_embedding_model()


# =========================================================
# 7. 벡터 DB 위치
# =========================================================

VECTOR_FOLDER = "vector_db"

INDEX_PATH = os.path.join(
    VECTOR_FOLDER,
    "index.faiss"
)

DOCUMENTS_PATH = os.path.join(
    VECTOR_FOLDER,
    "documents.pkl"
)


# =========================================================
# 8. 벡터 DB 확인
# =========================================================

if not os.path.exists(INDEX_PATH):

    st.error(
        "❌ vector_db/index.faiss 파일이 없습니다."
    )

    st.info(
        "vector_db 폴더를 GitHub에 업로드해주세요."
    )

    st.stop()


if not os.path.exists(DOCUMENTS_PATH):

    st.error(
        "❌ vector_db/documents.pkl 파일이 없습니다."
    )

    st.stop()


# =========================================================
# 9. 벡터 DB 불러오기
# =========================================================

@st.cache_resource
def load_vector_database():

    index = faiss.read_index(
        INDEX_PATH
    )

    with open(
        DOCUMENTS_PATH,
        "rb"
    ) as f:

        documents = pickle.load(f)

    return index, documents


with st.spinner(
    "📚 규정집 검색 DB를 불러오는 중..."
):

    vector_index, documents = (
        load_vector_database()
    )


# =========================================================
# 10. 규정집 목록 만들기
# =========================================================

filenames = []


for document in documents:

    filename = str(
        document.get("filename")
        or "파일명 없음"
    )

    if filename not in filenames:

        filenames.append(filename)


# =========================================================
# 11. 세션 상태
# =========================================================

if "selected_file" not in st.session_state:

    st.session_state.selected_file = None


if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================================
# 12. 첫 화면 안내
# =========================================================

if st.session_state.selected_file is None:

    st.markdown(
        """
        <div class="info-box">

            <div class="info-title">
                🤖 안녕하십니까.
            </div>

            <div class="info-text">

                저는 <b>대구경북지사 기술업무 담당 AI 챗봇</b>입니다.
                <br><br>

                등록된 기술업무 규정집을 검색하여
                질문과 관련성이 높은 내용을 찾아 답변해 드립니다.

                <br><br>

                <b>
                ※ 선택한 규정집에 명시되지 않은 내용은
                임의로 답변하지 않습니다.
                </b>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 13. 규정집 선택 화면
# =========================================================

if st.session_state.selected_file is None:

    st.markdown(
        '<div class="list-title">📚 질문할 규정집을 선택하세요.</div>',
        unsafe_allow_html=True
    )


    for filename in filenames:

        if st.button(
            f"📋 {filename}",
            key=f"select_{filename}",
            use_container_width=True
        ):

            st.session_state.selected_file = filename

            st.session_state.messages = []

            st.rerun()


# =========================================================
# 14. 선택된 규정집 화면
# =========================================================

else:

    selected_file = st.session_state.selected_file


    # ---------------------------------------------
    # 선택된 규정집 표시
    # ---------------------------------------------

    st.markdown(
        f"""
        <div class="selected-box">

            <div class="selected-title">
                📚 현재 질문할 규정집
            </div>

            <div class="selected-name">
                {selected_file}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ---------------------------------------------
    # 규정집 목록으로 돌아가기
    # ---------------------------------------------

    if st.button(
        "📚 규정집 목록",
        key="back_to_list",
        use_container_width=True
    ):

        st.session_state.selected_file = None

        st.session_state.messages = []

        st.rerun()


    st.markdown("")


# =========================================================
# 15. 검색 함수
# =========================================================

def search_documents(
    query,
    selected_filename,
    top_k=6
):

    if query is None:

        return []


    query = str(query).strip()


    if not query:

        return []


    # ---------------------------------------------
    # 질문 벡터화
    # ---------------------------------------------

    query_embedding = embedding_model.encode(
        ["query: " + query],
        normalize_embeddings=True
    )


    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )


    # ---------------------------------------------
    # FAISS 검색
    # ---------------------------------------------

    search_k = min(
        max(top_k * 15, 50),
        len(documents)
    )


    scores, indices = vector_index.search(
        query_embedding,
        search_k
    )


    results = []


    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:

            continue


        if int(idx) >= len(documents):

            continue


        result = documents[int(idx)].copy()


        filename = str(
            result.get("filename")
            or ""
        )


        # 선택한 규정집만 사용
        if filename != selected_filename:

            continue


        page = str(
            result.get("page")
            or "페이지 정보 없음"
        )


        text = str(
            result.get("text")
            or ""
        )


        if not text.strip():

            continue


        result["filename"] = filename

        result["page"] = page

        result["text"] = text

        result["score"] = float(score)


        results.append(result)


        if len(results) >= top_k:

            break


    return results


# =========================================================
# 16. 선택된 규정집이 있을 때만 채팅 표시
# =========================================================

if st.session_state.selected_file is not None:


    # ---------------------------------------------
    # 기존 대화 표시
    # ---------------------------------------------

    for message in st.session_state.messages:

        if message["role"] == "user":

            with st.chat_message("user"):

                st.markdown(
                    message["content"]
                )


        else:

            with st.chat_message("assistant"):

                st.markdown(
                    message["content"]
                )


    # ---------------------------------------------
    # 질문 입력창
    # ---------------------------------------------

    user_input = st.chat_input(
        "선택한 규정집에 대해 질문하세요."
    )


    # =====================================================
    # 17. 질문 처리
    # =====================================================

    if user_input:

        # ---------------------------------------------
        # 사용자 질문 저장
        # ---------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )


        # ---------------------------------------------
        # 사용자 질문 표시
        # ---------------------------------------------

        with st.chat_message("user"):

            st.markdown(
                user_input
            )


        # ---------------------------------------------
        # AI 답변
        # ---------------------------------------------

        with st.chat_message("assistant"):

            status = st.status(
                "📚 관련 규정을 검색하고 있습니다...",
                expanded=True
            )


            try:

                # =====================================
                # ① 선택된 규정집 검색
                # =====================================

                search_results = search_documents(
                    user_input,
                    st.session_state.selected_file,
                    top_k=6
                )


                # =====================================
                # ② 검색 결과 없음
                # =====================================

                if not search_results:

                    status.update(
                        label="❌ 관련 규정을 찾지 못했습니다.",
                        state="complete",
                        expanded=False
                    )


                    answer = (
                        "해당 내용은 선택한 규정집에 "
                        "명시되어 있지 않습니다."
                    )


                    st.markdown(answer)


                else:

                    # =================================
                    # ③ 검색 완료
                    # =================================

                    status.update(
                        label=(
                            f"✅ 관련 규정 "
                            f"{len(search_results)}건을 찾았습니다."
                        ),
                        state="running",
                        expanded=True
                    )


                    # =================================
                    # ④ 검색 결과 만들기
                    # =================================

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

내용:
{result["text"]}
"""
                        )


                    search_context = "\n".join(
                        context_parts
                    )


                    # =================================
                    # ⑤ Gemini 프롬프트
                    # =================================

                    prompt = f"""
너는 주택관리공단 대구경북지사의
기술업무 담당 AI 챗봇이다.

사용자가 선택한 규정집에서 검색된 내용만을
근거로 답변해야 한다.

[선택된 규정집]

{st.session_state.selected_file}

[답변 원칙]

1. 검색된 규정 문서의 내용을 근거로 답변한다.

2. 검색된 내용에 없는 사항은 추측하거나
만들어내지 않는다.

3. 일반적인 지식이나 인터넷 정보를 사용하지 않는다.

4. 답변 근거가 부족한 경우 다음과 같이 답변한다.

"해당 내용은 선택한 규정집에 명시되어 있지 않습니다."

5. 답변의 근거가 되는 문서명과 페이지를 표시한다.

6. 규정이나 업무절차는 이해하기 쉽게
번호 또는 항목으로 정리한다.

7. 검색된 내용이 서로 다른 경우
임의로 하나를 선택하지 말고 차이를 설명한다.

8. 답변은 한국어로 작성한다.

━━━━━━━━━━━━━━━━━━━━━━

[검색된 규정 내용]

{search_context}

━━━━━━━━━━━━━━━━━━━━━━

[사용자 질문]

{user_input}

━━━━━━━━━━━━━━━━━━━━━━

[답변]
"""


                    # =================================
                    # ⑥ Gemini 호출
                    # =================================

                    status.update(
                        label=(
                            "🤖 관련 규정을 확인했습니다. "
                            "답변을 작성하고 있습니다..."
                        ),
                        state="running",
                        expanded=True
                    )


                    answer = None


                    for attempt in range(3):

                        try:

                            response = (
                                client.models.generate_content(
                                    model="gemini-3.6-flash",
                                    contents=prompt
                                )
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

                                    time.sleep(
                                        2 ** attempt
                                    )

                                    continue


                            answer = (
                                "⚠️ Gemini AI 오류가 발생했습니다.\n\n"
                                f"{error_text}"
                            )

                            break


                    if not answer:

                        answer = (
                            "⚠️ 현재 AI 서버가 "
                            "응답하지 않습니다.\n\n"
                            "잠시 후 다시 질문해 주세요."
                        )


                    # =================================
                    # ⑦ 답변 완료
                    # =================================

                    status.update(
                        label="✅ 답변 작성이 완료되었습니다.",
                        state="complete",
                        expanded=False
                    )


                    st.markdown(answer)


                    # =================================
                    # ⑧ 답변 근거
                    # =================================

                    st.markdown("---")

                    st.markdown(
                        "### 📚 답변 근거"
                    )


                    shown_sources = set()


                    for result in search_results:

                        source_key = (
                            result["filename"],
                            result["page"]
                        )


                        if source_key in shown_sources:

                            continue


                        shown_sources.add(
                            source_key
                        )


                        st.markdown(
                            f"📋 **{result['filename']}**  \n"
                            f"📖 {result['page']}페이지"
                        )


                # =====================================
                # ⑨ 답변 저장
                # =====================================

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )


            except Exception as e:

                status.update(
                    label="❌ 검색 중 오류가 발생했습니다.",
                    state="error",
                    expanded=True
                )


                error_answer = (
                    "⚠️ 검색 중 오류가 발생했습니다.\n\n"
                    f"`{str(e)}`"
                )


                st.markdown(
                    error_answer
                )


                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_answer
                    }
                )
