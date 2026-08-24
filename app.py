import streamlit as st
from google import genai

import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer

import os
import time
import html


# =========================================================
# 1. 페이지 설정
# =========================================================

st.set_page_config(
    page_title="대구경북 AI 챗봇",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# =========================================================
# 2. 전체 디자인 CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       전체 화면
    ===================================================== */

    .stApp {
        background: #f5f8fc;
    }

    .main .block-container {
        max-width: 900px;
        padding-top: 10px;
        padding-bottom: 80px;
    }


    /* =====================================================
       상단 파란색 헤더
    ===================================================== */

    .hero {

        background: linear-gradient(
            135deg,
            #0756c9 0%,
            #1976e8 55%,
            #48a1f5 100%
        );

        border-radius: 0 0 24px 24px;

        padding: 18px 24px;

        color: white;

        margin-bottom: 18px;

        box-shadow:
            0 6px 18px rgba(0, 82, 180, 0.16);
    }


    /* 한 줄 배치 */

    .hero-top {

        display: flex;

        align-items: center;

        gap: 10px;

        min-height: 50px;
    }


    /* 로고 */

    .hero-logo {

        width: 46px;
        height: 46px;

        border-radius: 13px;

        background: rgba(255,255,255,0.96);

        display: flex;

        align-items: center;

        justify-content: center;

        font-size: 26px;

        flex-shrink: 0;

        box-shadow:
            0 3px 10px rgba(0,0,0,0.12);
    }


    /* 주택관리공단 */

    .hero-title-small {

        font-size: 16px;

        font-weight: 800;

        line-height: 1.25;

        white-space: nowrap;
    }


    .hero-title-small span {

        display: block;

        font-size: 11px;

        margin-top: 2px;

        opacity: 0.9;
    }


    /* 구분선 */

    .hero-divider {

        font-size: 24px;

        opacity: 0.6;

        margin-left: 4px;
    }


    /* 기술업무 AI 챗봇 */

    .hero-chat-title {

        font-size: 25px;

        font-weight: 800;

        letter-spacing: -1px;

        white-space: nowrap;
    }


    /* =====================================================
       안내 카드
    ===================================================== */

    .info-card {

        background: white;

        border-radius: 22px;

        padding: 22px 24px;

        margin-bottom: 16px;

        border: 1px solid #e1e8f0;

        box-shadow:
            0 5px 20px rgba(30,70,110,0.08);
    }


    .info-title {

        font-size: 20px;

        font-weight: 800;

        color: #17365f;

        margin-bottom: 12px;
    }


    .info-text {

        color: #4c5d72;

        line-height: 1.75;

        font-size: 15px;
    }


    /* =====================================================
       주의 문구
    ===================================================== */

    .notice-text {

        font-size: 14px;

        font-weight: 700;

        color: #243d5a;

        margin: 18px 3px 10px 3px;
    }


    /* =====================================================
       선택된 규정집
    ===================================================== */

    .selected-card {

        background: linear-gradient(
            135deg,
            #e9f7ff,
            #f5fbff
        );

        border: 1px solid #c6e4fa;

        border-radius: 18px;

        padding: 18px 20px;

        margin-top: 10px;

        margin-bottom: 12px;

        box-shadow:
            0 4px 12px rgba(30,100,160,0.06);
    }


    .selected-label {

        font-size: 13px;

        color: #3174ae;

        font-weight: 700;

        margin-bottom: 6px;
    }


    .selected-file {

        font-size: 17px;

        color: #163b68;

        font-weight: 800;

        word-break: keep-all;

        line-height: 1.5;
    }


    /* =====================================================
       질문 말풍선
    ===================================================== */

    .user-bubble {

        background: #1263d6;

        color: white;

        border-radius: 20px 20px 5px 20px;

        padding: 15px 18px;

        margin: 18px 0 10px auto;

        max-width: 82%;

        font-size: 15px;

        line-height: 1.6;

        box-shadow:
            0 5px 15px rgba(18,99,214,0.18);
    }


    /* =====================================================
       AI 답변 카드
    ===================================================== */

    .ai-card {

        background: white;

        border-radius: 20px;

        border: 1px solid #e2e9f2;

        padding: 20px;

        margin: 10px 0 18px 0;

        box-shadow:
            0 5px 18px rgba(30,70,110,0.07);
    }


    .ai-header {

        display: flex;

        align-items: center;

        gap: 10px;

        font-weight: 800;

        color: #174d8f;

        margin-bottom: 12px;
    }


    .ai-icon {

        width: 38px;

        height: 38px;

        border-radius: 50%;

        background: #eef6ff;

        display: flex;

        align-items: center;

        justify-content: center;

        font-size: 22px;
    }


    .ai-content {

        color: #26384c;

        font-size: 15px;

        line-height: 1.8;
    }


    /* =====================================================
       답변 근거
    ===================================================== */

    .source-card {

        background: #eefaf8;

        border: 1px solid #cdebe5;

        border-radius: 14px;

        padding: 13px 15px;

        margin-top: 10px;

        color: #245c57;

        font-size: 13px;

        line-height: 1.6;
    }


    /* =====================================================
       버튼
    ===================================================== */

    div.stButton > button {

        border-radius: 14px !important;

        border: 1px solid #c7ddf7 !important;

        background: white !important;

        color: #175ca8 !important;

        font-weight: 700 !important;

        min-height: 44px !important;

        transition: all 0.2s ease;
    }


    div.stButton > button:hover {

        border-color: #2677d9 !important;

        background: #eef6ff !important;

        color: #0756c9 !important;
    }


    /* =====================================================
       규정집 목록 제목
    ===================================================== */

    .document-title {

        font-size: 19px;

        font-weight: 800;

        color: #17365f;

        margin-top: 12px;

        margin-bottom: 10px;
    }


    /* =====================================================
       하단 Streamlit footer 제거
    ===================================================== */

    footer {
        visibility: hidden;
    }


    /* 상단 GitHub / Share / 메뉴 등 툴바 숨기기 */

    [data-testid="stToolbar"] {
        visibility: hidden;
        height: 0;
    }


    /* =====================================================
       모바일 화면
    ===================================================== */

    @media (max-width: 600px) {

        .main .block-container {

            padding-left: 12px;
            padding-right: 12px;

            padding-top: 5px;

            padding-bottom: 70px;
        }


        .hero {

            border-radius: 0 0 22px 22px;

            padding: 15px 16px;

            margin-bottom: 14px;
        }


        .hero-top {

            gap: 7px;
        }


        .hero-logo {

            width: 40px;

            height: 40px;

            font-size: 23px;
        }


        .hero-title-small {

            font-size: 13px;
        }


        .hero-title-small span {

            font-size: 9px;
        }


        .hero-divider {

            font-size: 20px;
        }


        .hero-chat-title {

            font-size: 18px;

            letter-spacing: -1px;
        }


        .info-card {

            padding: 19px;

            border-radius: 19px;
        }


        .info-title {

            font-size: 18px;
        }


        .info-text {

            font-size: 14px;
        }


        .selected-file {

            font-size: 15px;
        }


        .user-bubble {

            max-width: 90%;

            font-size: 14px;
        }


        .ai-content {

            font-size: 14px;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 3. 상단 메인 헤더
# =========================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-top">

            <div class="hero-logo">
                📚
            </div>

            <div class="hero-title-small">
                주택관리공단
                <span>대구경북지사</span>
            </div>

            <div class="hero-divider">
                |
            </div>

            <div class="hero-chat-title">
                기술업무 AI 챗봇
            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 4. Gemini API 확인
# =========================================================

api_key = st.secrets.get("GEMINI_API_KEY")


if not api_key:

    st.error(
        "Gemini API 키가 설정되지 않았습니다.\n\n"
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
        "관리자 PC에서 build_vector_db.py를 실행한 후 "
        "생성된 vector_db 폴더를 GitHub에 업로드해주세요."
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
# 10. 등록된 PDF 목록 만들기
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
# 12. 검색 함수
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


    if not selected_filename:

        return []


    # -----------------------------------------------------
    # 질문을 벡터로 변환
    # -----------------------------------------------------

    query_embedding = embedding_model.encode(
        ["query: " + query],
        normalize_embeddings=True
    )


    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )


    # -----------------------------------------------------
    # FAISS 검색
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
            or ""
        )


        # -------------------------------------------------
        # 선택된 PDF와 같은 문서인지 확인
        # -------------------------------------------------

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
# 13. 첫 화면 - 규정집 선택 화면
# =========================================================

if not st.session_state.selected_file:

    # -----------------------------------------------------
    # 안내 카드
    # -----------------------------------------------------

    st.markdown(
        """
        <div class="info-card">

            <div class="info-title">
                🤖 안녕하십니까.
            </div>

            <div class="info-text">

                저는 <b>대구경북지사 기술업무 담당 AI 챗봇</b>입니다.<br><br>

                등록된 기술업무 규정집을 검색하여
                질문과 관련성이 높은 내용을 찾아 답변해 드립니다.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # 주의사항
    # -----------------------------------------------------

    st.markdown(
        """
        <div class="notice-text">
            ※ 선택한 규정집에 명시되지 않은 내용은
            임의로 답변하지 않습니다.
        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # 규정집 목록 제목
    # -----------------------------------------------------

    st.markdown(
        """
        <div class="document-title">
            📚 규정집 목록
        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # 모든 규정집 버튼 표시
    # -----------------------------------------------------

    for filename in filenames:

        if st.button(
            f"📋 {filename}",
            key=f"document_{filename}",
            use_container_width=True
        ):

            # 선택한 규정집 저장
            st.session_state.selected_file = filename

            # 기존 대화 삭제
            st.session_state.messages = []

            # 화면 새로고침
            st.rerun()


# =========================================================
# 14. 규정집 선택 후 - 질문 화면
# =========================================================

else:

    selected_file = st.session_state.selected_file


    # -----------------------------------------------------
    # 선택한 규정집 표시
    # -----------------------------------------------------

    safe_selected_file = html.escape(
        selected_file
    )


    st.markdown(
        f"""
        <div class="selected-card">

            <div class="selected-label">
                📚 현재 선택한 규정집
            </div>

            <div class="selected-file">
                {safe_selected_file}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # 규정집 목록으로 돌아가기 버튼
    # -----------------------------------------------------

    if st.button(
        "📚 규정집 목록으로 돌아가기",
        use_container_width=True
    ):

        st.session_state.selected_file = None

        st.session_state.messages = []

        st.rerun()


    # -----------------------------------------------------
    # 현재 선택한 규정집 안내
    # -----------------------------------------------------

    st.markdown(
        """
        <div class="notice-text">
            🔎 현재 선택한 규정집의 내용만 검색하여 답변합니다.
        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # 기존 대화 표시
    # =====================================================

    for message in st.session_state.messages:

        if message["role"] == "user":

            safe_content = html.escape(
                str(message["content"])
            )

            st.markdown(
                f"""
                <div class="user-bubble">
                    {safe_content}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="ai-card">

                    <div class="ai-header">

                        <div class="ai-icon">
                            🤖
                        </div>

                        <div>
                            대구경북 기술업무 AI
                        </div>

                    </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                message["content"]
            )

            st.markdown(
                """
                </div>
                """,
                unsafe_allow_html=True
            )


    # =====================================================
    # 질문 입력창
    # =====================================================

    user_input = st.chat_input(
        "선택한 규정집에 대해 질문하세요."
    )


    # =====================================================
    # 질문 처리
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
        # 사용자 질문 표시
        # -------------------------------------------------

        safe_user_input = html.escape(
            user_input
        )


        st.markdown(
            f"""
            <div class="user-bubble">
                {safe_user_input}
            </div>
            """,
            unsafe_allow_html=True
        )


        # =================================================
        # AI 검색 및 답변
        # =================================================

        with st.spinner(
            "📚 선택한 규정집에서 관련 내용을 검색하고 있습니다..."
        ):

            try:

                # -----------------------------------------
                # ① 선택한 PDF에서 검색
                # -----------------------------------------

                search_results = search_documents(
                    user_input,
                    selected_file,
                    top_k=6
                )


                # -----------------------------------------
                # ② 검색 결과가 없는 경우
                # -----------------------------------------

                if not search_results:

                    answer = (
                        "해당 내용은 선택한 규정집에 "
                        "명시되어 있지 않습니다."
                    )


                else:

                    # -------------------------------------
                    # ③ 검색 결과 정리
                    # -------------------------------------

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


                    # -------------------------------------
                    # ④ Gemini 프롬프트
                    # -------------------------------------

                    prompt = f"""
너는 주택관리공단 대구경북지사의
기술업무 담당 AI 챗봇이다.

사용자가 선택한 규정집의 검색 결과만을
근거로 답변해야 한다.

━━━━━━━━━━━━━━━━━━━━━━
[선택된 규정집]
━━━━━━━━━━━━━━━━━━━━━━

{selected_file}

━━━━━━━━━━━━━━━━━━━━━━
[답변 원칙]
━━━━━━━━━━━━━━━━━━━━━━

1. 반드시 검색된 규정 문서의 내용을 근거로 답변한다.

2. 검색된 내용에 없는 사항은 추측하거나 만들어내지 않는다.

3. 일반적인 지식이나 인터넷 정보를 사용하지 않는다.

4. 답변 근거가 부족하면 다음 문장을 사용한다.

"해당 내용은 선택한 규정집에 명시되어 있지 않습니다."

5. 답변에는 관련 문서명과 페이지를 표시한다.

6. 규정이나 절차는 이해하기 쉽도록 항목별로 정리한다.

7. 검색된 문서에 서로 다른 내용이 있으면
임의로 판단하지 말고 차이를 설명한다.

8. 답변은 한국어로 작성한다.

━━━━━━━━━━━━━━━━━━━━━━
[검색된 규정 내용]
━━━━━━━━━━━━━━━━━━━━━━

{search_context}

━━━━━━━━━━━━━━━━━━━━━━
[사용자 질문]
━━━━━━━━━━━━━━━━━━━━━━

{user_input}

━━━━━━━━━━━━━━━━━━━━━━
[답변]
━━━━━━━━━━━━━━━━━━━━━━
"""


                    # -------------------------------------
                    # ⑤ Gemini 호출
                    # -------------------------------------

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


                # =================================================
                # ⑥ AI 답변 표시
                # =================================================

                st.markdown(
                    """
                    <div class="ai-card">

                        <div class="ai-header">

                            <div class="ai-icon">
                                🤖
                            </div>

                            <div>
                                대구경북 기술업무 AI
                            </div>

                        </div>

                    """,
                    unsafe_allow_html=True
                )


                # 답변은 Markdown으로 표시
                # → 번호, 굵은 글씨, 표 등을 정상적으로 표시
                st.markdown(
                    answer
                )


                st.markdown(
                    """
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # =================================================
                # ⑦ 답변 근거
                # =================================================

                if search_results:

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


                        safe_filename = html.escape(
                            result["filename"]
                        )

                        safe_page = html.escape(
                            str(result["page"])
                        )


                        st.markdown(
                            f"""
                            <div class="source-card">

                                📄 <b>답변 근거</b><br>

                                📋 {safe_filename}<br>

                                📖 {safe_page}페이지

                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                # =================================================
                # ⑧ 답변 저장
                # =================================================

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )


            # =================================================
            # 오류 처리
            # =================================================

            except Exception as e:

                error_answer = (
                    "⚠️ 검색 중 오류가 발생했습니다.\n\n"
                    f"{str(e)}"
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
