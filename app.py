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
# 2. 화면 디자인 CSS
# =========================================================

st.markdown(
    """
<style>

/* ---------------------------------------------------------
   전체 배경
--------------------------------------------------------- */

.stApp {
    background: #f5f8fc;
}


/* ---------------------------------------------------------
   상단 Streamlit 기본 영역 숨김
--------------------------------------------------------- */

header {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* ---------------------------------------------------------
   전체 컨테이너
--------------------------------------------------------- */

.block-container {
    max-width: 850px;
    padding-top: 0.5rem !important;
    padding-bottom: 100px !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}


/* ---------------------------------------------------------
   메인 헤더
--------------------------------------------------------- */

.hero {
    background:
        linear-gradient(
            135deg,
            #064dbb 0%,
            #1769d8 55%,
            #3288e8 100%
        );

    border-radius: 0 0 30px 30px;

    padding: 28px 25px 35px 25px;

    color: white;

    margin-bottom: 20px;

    box-shadow:
        0 8px 25px rgba(0, 70, 160, 0.18);
}


.hero-top {
    display: flex;
    align-items: center;
    gap: 10px;

    font-size: 15px;
    font-weight: 600;

    margin-bottom: 28px;
}


.logo-box {
    width: 42px;
    height: 42px;

    border-radius: 12px;

    background: rgba(255,255,255,0.15);

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 25px;
}


.hero-title {
    font-size: 34px;
    font-weight: 800;

    line-height: 1.25;

    margin-bottom: 12px;
}


.hero-subtitle {
    font-size: 16px;
    line-height: 1.75;

    opacity: 0.95;
}


/* ---------------------------------------------------------
   로봇
--------------------------------------------------------- */

.robot-area {
    display: flex;
    justify-content: flex-end;

    margin-top: -25px;
    margin-bottom: -5px;
}


.robot {
    width: 105px;
    height: 105px;

    border-radius: 50%;

    background: rgba(255,255,255,0.16);

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 65px;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.12);
}


/* ---------------------------------------------------------
   안내 카드
--------------------------------------------------------- */

.info-card {

    background: white;

    border-radius: 22px;

    padding: 22px;

    margin-bottom: 18px;

    box-shadow:
        0 4px 18px rgba(20, 60, 100, 0.08);

    border: 1px solid #e5ebf3;
}


.info-title {
    font-size: 17px;
    font-weight: 800;

    color: #1c3557;

    margin-bottom: 10px;
}


.info-text {
    font-size: 14px;
    line-height: 1.75;

    color: #4b5870;
}


/* ---------------------------------------------------------
   자주 묻는 질문
--------------------------------------------------------- */

.quick-title {

    font-size: 17px;

    font-weight: 800;

    color: #172b4d;

    margin: 25px 3px 12px 3px;
}


/* ---------------------------------------------------------
   선택된 규정집 카드
--------------------------------------------------------- */

.selected-card {

    background:
        linear-gradient(
            135deg,
            #ffffff,
            #eef6ff
        );

    border: 1px solid #c9ddf7;

    border-radius: 20px;

    padding: 18px 20px;

    margin-bottom: 15px;
}


.selected-label {

    font-size: 13px;

    color: #3774bd;

    font-weight: 700;

    margin-bottom: 6px;
}


.selected-file {

    font-size: 16px;

    color: #18395e;

    font-weight: 800;

    line-height: 1.5;

    word-break: keep-all;
}


/* ---------------------------------------------------------
   AI 메시지
--------------------------------------------------------- */

.ai-message {

    display: flex;

    align-items: flex-start;

    gap: 10px;

    margin: 14px 0;
}


.ai-icon {

    min-width: 42px;
    height: 42px;

    border-radius: 50%;

    background: #e7f2ff;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 23px;

    border: 1px solid #c9e1fb;
}


.ai-bubble {

    background: white;

    border-radius: 4px 18px 18px 18px;

    padding: 15px 17px;

    color: #26384f;

    font-size: 15px;

    line-height: 1.75;

    box-shadow:
        0 3px 14px rgba(30,70,110,0.07);

    border: 1px solid #e5ebf2;

    flex: 1;
}


/* ---------------------------------------------------------
   사용자 메시지
--------------------------------------------------------- */

.user-message {

    display: flex;

    justify-content: flex-end;

    margin: 18px 0;
}


.user-bubble {

    max-width: 82%;

    background:
        linear-gradient(
            135deg,
            #0759ca,
            #287be1
        );

    color: white;

    border-radius: 18px 18px 4px 18px;

    padding: 12px 17px;

    font-size: 15px;

    line-height: 1.6;

    box-shadow:
        0 4px 12px rgba(0,80,180,0.15);
}


/* ---------------------------------------------------------
   답변 근거 카드
--------------------------------------------------------- */

.source-card {

    background: #e9f7f5;

    border: 1px solid #c7ebe5;

    border-radius: 12px;

    padding: 13px 15px;

    margin-top: 13px;

    color: #235d58;

    font-size: 13px;

    line-height: 1.6;
}


.source-title {

    font-weight: 800;

    color: #196b64;

    margin-bottom: 4px;
}


/* ---------------------------------------------------------
   규정집 버튼
--------------------------------------------------------- */

div.stButton > button {

    width: 100%;

    border-radius: 14px;

    border: 1px solid #c9ddf5;

    background: white;

    color: #24476f;

    font-weight: 700;

    min-height: 48px;

    transition: 0.2s;
}


div.stButton > button:hover {

    border-color: #1769d8;

    color: #1769d8;

    background: #f2f8ff;
}


/* ---------------------------------------------------------
   목록 버튼
--------------------------------------------------------- */

.back-button button {

    background: #edf4fc !important;

    border: 1px solid #c5d9ef !important;

    color: #31577f !important;

    font-weight: 700 !important;
}


/* ---------------------------------------------------------
   입력창
--------------------------------------------------------- */

.stChatInput {

    bottom: 15px;
}


.stChatInputContainer {

    border-radius: 20px !important;
}


/* ---------------------------------------------------------
   모바일
--------------------------------------------------------- */

@media (max-width: 600px) {

    .block-container {

        padding-left: 10px !important;
        padding-right: 10px !important;

        padding-top: 0 !important;
    }


    .hero {

        border-radius: 0 0 25px 25px;

        padding: 23px 18px 28px 18px;
    }


    .hero-title {

        font-size: 30px;
    }


    .hero-subtitle {

        font-size: 14px;
    }


    .robot {

        width: 85px;
        height: 85px;

        font-size: 52px;
    }


    .info-card {

        padding: 17px;

        border-radius: 18px;
    }


    .ai-bubble {

        font-size: 14px;
    }


    .user-bubble {

        font-size: 14px;
    }

}


/* ---------------------------------------------------------
   Streamlit 기본 링크 아이콘 숨김
--------------------------------------------------------- */

[data-testid="stMarkdownContainer"] a {

    text-decoration: none;
}


/* ---------------------------------------------------------
   상태창 최소화
--------------------------------------------------------- */

[data-testid="stStatusWidget"] {

    border-radius: 12px;
}


/* ---------------------------------------------------------
   스크롤바
--------------------------------------------------------- */

::-webkit-scrollbar {

    width: 6px;
}


::-webkit-scrollbar-thumb {

    background: #cbd7e6;

    border-radius: 10px;
}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# 3. Gemini API
# =========================================================

api_key = st.secrets.get("GEMINI_API_KEY")


if not api_key:

    st.error(
        "Gemini API 키가 설정되지 않았습니다."
    )

    st.stop()


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
# 4. 임베딩 모델
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
# 5. 벡터 DB
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


if not os.path.exists(INDEX_PATH):

    st.error(
        "❌ vector_db/index.faiss 파일이 없습니다."
    )

    st.stop()


if not os.path.exists(DOCUMENTS_PATH):

    st.error(
        "❌ vector_db/documents.pkl 파일이 없습니다."
    )

    st.stop()


# =========================================================
# 6. 벡터 DB 불러오기
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
# 7. 세션 상태
# =========================================================

if "selected_file" not in st.session_state:

    st.session_state.selected_file = None


if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================================
# 8. 등록된 PDF 목록
# =========================================================

filenames = []

for document in documents:

    filename = str(
        document.get("filename") or "파일명 없음"
    )

    if filename not in filenames:

        filenames.append(filename)


# =========================================================
# 9. 벡터 검색 함수
# =========================================================

def search_documents(
    query,
    selected_file=None,
    top_k=6
):

    if not query:

        return []


    query = str(query).strip()


    if not query:

        return []


    # 질문을 벡터로 변환

    query_embedding = embedding_model.encode(
        ["query: " + query],
        normalize_embeddings=True
    )


    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )


    # -----------------------------------------------------
    # 선택한 PDF가 있는 경우
    # 충분히 많이 검색한 후 선택 문서만 필터링
    # -----------------------------------------------------

    search_k = min(
        max(top_k * 10, 30),
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
            or "파일명 없음"
        )


        # 선택한 규정집만 검색

        if selected_file:

            if filename != selected_file:

                continue


        text = str(
            result.get("text")
            or ""
        )


        if not text.strip():

            continue


        page = str(
            result.get("page")
            or "페이지 정보 없음"
        )


        result["filename"] = filename
        result["page"] = page
        result["text"] = text
        result["score"] = float(score)


        results.append(result)


        if len(results) >= top_k:

            break


    return results


# =========================================================
# 10. 메인 헤더
# =========================================================

st.markdown(
    """
<div class="hero">

    <div class="hero-top">

        <div class="logo-box">
            🏢
        </div>

        <div>
            주택관리공단<br>
            <span style="font-size:12px;">
                대구경북지사
            </span>
        </div>

    </div>


    <div class="hero-title">
        기술업무 AI 챗봇
    </div>


    <div class="hero-subtitle">

        기술업무 관련 규정·지침을<br>
        문서에 근거하여 정확하게<br>
        답변해 드립니다.

    </div>


    <div class="robot-area">

        <div class="robot">
            🤖
        </div>

    </div>

</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# 11. 규정집 선택 화면
# =========================================================

if st.session_state.selected_file is None:

    st.markdown(
        """
<div class="info-card">

    <div class="info-title">
        🤖 안녕하십니까.
    </div>

    <div class="info-text">

        저는 <b>대구경북지사 기술업무 담당 AI 챗봇</b>입니다.<br><br>

        아래 규정집을 선택하시면<br>
        해당 문서의 내용만 검색하여 답변해 드립니다.

    </div>

</div>
""",
        unsafe_allow_html=True
    )


    st.markdown(
        """
<div class="quick-title">
    📚 질문할 규정집을 선택하세요
</div>
""",
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # 규정집 버튼
    # -----------------------------------------------------

    for i, filename in enumerate(filenames):

        if st.button(
            f"📋  {filename}",
            key=f"pdf_{i}"
        ):

            st.session_state.selected_file = filename

            st.session_state.messages = []

            st.rerun()


    st.markdown(
        """
<div style="
    margin-top:20px;
    padding:15px;
    text-align:center;
    color:#718096;
    font-size:13px;
">

    💡 규정집을 선택한 후 해당 문서에 대해 질문할 수 있습니다.

</div>
""",
        unsafe_allow_html=True
    )


# =========================================================
# 12. 선택된 규정집 화면
# =========================================================

else:

    selected_file = (
        st.session_state.selected_file
    )


    # -----------------------------------------------------
    # 선택된 문서 표시
    # -----------------------------------------------------

    st.markdown(
        f"""
<div class="selected-card">

    <div class="selected-label">
        📚 현재 선택된 규정집
    </div>

    <div class="selected-file">
        {selected_file}
    </div>

</div>
""",
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # 최초 안내 메시지
    # -----------------------------------------------------

    if not st.session_state.messages:

        welcome = f"""
안녕하십니까.

현재 **「{selected_file}」** 규정집을 선택하셨습니다.

이 규정집의 내용만 검색하여 답변해 드립니다.

궁금하신 사항을 질문해 주세요.

※ 선택한 규정집에 명시되지 않은 내용은
임의로 답변하지 않습니다.
"""


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": welcome
            }
        )


    # -----------------------------------------------------
    # 기존 대화 출력
    # -----------------------------------------------------

    for message in st.session_state.messages:

        if message["role"] == "user":

            st.markdown(
                f"""
<div class="user-message">

    <div class="user-bubble">

        {message["content"]}

    </div>

</div>
""",
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
<div class="ai-message">

    <div class="ai-icon">
        🤖
    </div>

    <div class="ai-bubble">

        {message["content"]}

    </div>

</div>
""",
                unsafe_allow_html=True
            )


    # -----------------------------------------------------
    # 규정집 목록 버튼
    # -----------------------------------------------------

    st.markdown(
        '<div class="back-button">',
        unsafe_allow_html=True
    )


    if st.button(
        "← 규정집 목록",
        key="back_to_list"
    ):

        st.session_state.selected_file = None

        st.session_state.messages = []

        st.rerun()


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    # =====================================================
    # 질문 입력
    # =====================================================

    user_input = st.chat_input(
        "질문을 입력하세요..."
    )


    # =====================================================
    # 질문이 입력되었을 때만 실행
    # =====================================================

    if user_input:

        # -------------------------------------------------
        # 사용자 질문 저장
        # -------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )


        # -------------------------------------------------
        # 화면에 사용자 질문
        # -------------------------------------------------

        st.markdown(
            f"""
<div class="user-message">

    <div class="user-bubble">

        {user_input}

    </div>

</div>
""",
            unsafe_allow_html=True
        )


        # -------------------------------------------------
        # AI 응답
        # -------------------------------------------------

        with st.spinner(
            "📚 관련 규정을 검색하고 있습니다..."
        ):

            try:

                # =========================================
                # ① 벡터 검색
                # =========================================

                search_results = search_documents(
                    user_input,
                    selected_file=selected_file,
                    top_k=6
                )


                # =========================================
                # ② 검색 결과 없음
                # =========================================

                if not search_results:

                    answer = (
                        "해당 내용은 선택한 규정집에 "
                        "명시되어 있지 않습니다."
                    )


                    st.markdown(
                        f"""
<div class="ai-message">

    <div class="ai-icon">
        🤖
    </div>

    <div class="ai-bubble">

        {answer}

    </div>

</div>
""",
                        unsafe_allow_html=True
                    )


                else:

                    # =====================================
                    # ③ 검색 결과를 Gemini에 전달
                    # =====================================

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
{result["page"]}

내용:
{result["text"]}
"""
                        )


                    search_context = "\n".join(
                        context_parts
                    )


                    # =====================================
                    # ④ Gemini 프롬프트
                    # =====================================

                    prompt = f"""
너는 주택관리공단 대구경북지사의
기술업무 담당 AI 챗봇이다.

현재 사용자가 선택한 문서는 다음과 같다.

[선택 문서]
{selected_file}

사용자의 질문에 대해
아래 [검색된 규정 내용]만을 근거로 답변해야 한다.

[답변 원칙]

1. 반드시 검색된 규정 내용만 근거로 답변한다.

2. 문서에 없는 내용을 추측하거나 만들어내지 않는다.

3. 답변 근거를 찾을 수 없는 경우 다음과 같이 답변한다.

"해당 내용은 선택한 규정집에 명시되어 있지 않습니다."

4. 답변에는 가능한 경우 문서명과 페이지를 표시한다.

5. 규정이나 절차는 이해하기 쉽게 항목별로 정리한다.

6. 인터넷 검색을 사용하지 않는다.

7. 일반적인 지식보다 검색된 규정 내용을 우선한다.

8. 검색된 내용끼리 서로 다른 경우
임의로 하나를 선택하지 말고 차이가 있음을 알려준다.

9. 답변은 한국어로 작성한다.

━━━━━━━━━━━━━━━━━━━━━━

[검색된 규정 내용]

{search_context}

━━━━━━━━━━━━━━━━━━━━━━

[사용자 질문]

{user_input}

━━━━━━━━━━━━━━━━━━━━━━

답변을 작성하라.
"""


                    # =====================================
                    # ⑤ Gemini 답변
                    # =====================================

                    answer = None


                    model_name = st.secrets.get(
                        "GEMINI_MODEL",
                        "gemini-3.6-flash"
                    )


                    for attempt in range(3):

                        try:

                            response = client.models.generate_content(
                                model=model_name,
                                contents=prompt
                            )


                            answer = (
                                response.text
                                if response
                                else None
                            )


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
                                "⚠️ AI 서버 오류가 발생했습니다.\n\n"
                                f"`{error_text}`"
                            )

                            break


                    # =====================================
                    # ⑥ 답변이 없는 경우
                    # =====================================

                    if not answer:

                        answer = (
                            "⚠️ 현재 AI 서버가 일시적으로 "
                            "응답하지 않습니다.\n\n"
                            "잠시 후 다시 질문해 주세요."
                        )


                    # =====================================
                    # ⑦ AI 답변 출력
                    # =====================================

                    st.markdown(
                        f"""
<div class="ai-message">

    <div class="ai-icon">
        🤖
    </div>

    <div class="ai-bubble">

        {answer}

    </div>

</div>
""",
                        unsafe_allow_html=True
                    )


                    # =====================================
                    # ⑧ 답변 근거
                    # =====================================

                    shown_sources = set()


                    source_html = ""


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


                        source_html += f"""
<div class="source-card">

    <div class="source-title">
        📄 답변 근거
    </div>

    <b>{result["filename"]}</b><br>

    📖 {result["page"]}페이지

</div>
"""


                    if source_html:

                        st.markdown(
                            source_html,
                            unsafe_allow_html=True
                        )


                # =========================================
                # ⑨ 답변 저장
                # =========================================

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )


            except Exception as e:

                error_answer = (
                    "⚠️ 검색 중 오류가 발생했습니다.\n\n"
                    f"`{str(e)}`"
                )


                st.error(
                    error_answer
                )


                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_answer
                    }
                )
