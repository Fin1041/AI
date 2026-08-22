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
# 2. 전체 디자인 CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ---------------------------------------------
       전체 배경
    --------------------------------------------- */

    .stApp {
        background: #f5f8fc;
    }

    .main .block-container {
        max-width: 900px;
        padding-top: 20px;
        padding-bottom: 100px;
    }


    /* ---------------------------------------------
       상단 헤더
    --------------------------------------------- */

    .hero {
        background: linear-gradient(
            135deg,
            #0756c9 0%,
            #1976e8 55%,
            #48a1f5 100%
        );

        border-radius: 0 0 28px 28px;

        padding: 28px 28px 32px 28px;

        color: white;

        margin-bottom: 18px;

        box-shadow:
            0 8px 25px rgba(0, 82, 180, 0.18);
    }


    .hero-top {
        display: flex;
        align-items: center;
        gap: 12px;

        font-size: 16px;
        font-weight: 700;

        margin-bottom: 28px;
    }


    .hero-logo {
        width: 52px;
        height: 52px;

        border-radius: 16px;

        background: rgba(255,255,255,0.95);

        display: flex;
        align-items: center;
        justify-content: center;

        font-size: 30px;

        box-shadow:
            0 4px 12px rgba(0,0,0,0.15);
    }


    .hero-title {
        font-size: 38px;
        font-weight: 800;

        letter-spacing: -1.5px;

        margin-bottom: 12px;
    }


    .hero-subtitle {
        font-size: 16px;
        line-height: 1.8;

        opacity: 0.95;
    }


    /* ---------------------------------------------
       안내 카드
    --------------------------------------------- */

    .info-card {

        background: white;

        border-radius: 22px;

        padding: 24px;

        margin-top: -5px;
        margin-bottom: 18px;

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


    /* ---------------------------------------------
       선택된 규정집
    --------------------------------------------- */

    .selected-card {

        background: linear-gradient(
            135deg,
            #e9f7ff,
            #f5fbff
        );

        border: 1px solid #c6e4fa;

        border-radius: 18px;

        padding: 18px 20px;

        margin-top: 15px;
        margin-bottom: 15px;
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
    }


    /* ---------------------------------------------
       질문 말풍선
    --------------------------------------------- */

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


    /* ---------------------------------------------
       AI 답변
    --------------------------------------------- */

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


    /* ---------------------------------------------
       답변 근거
    --------------------------------------------- */

    .source-card {

        background: #eefaf8;

        border: 1px solid #cdebe5;

        border-radius: 14px;

        padding: 13px 15px;

        margin-top: 15px;

        color: #245c57;

        font-size: 13px;

        line-height: 1.6;
    }


    /* ---------------------------------------------
       규정집 선택 버튼
    --------------------------------------------- */

    div.stButton > button {

        border-radius: 14px !important;

        border: 1px solid #c7ddf7 !important;

        background: white !important;

        color: #175ca8 !important;

        font-weight: 700 !important;

        min-height: 42px !important;

        transition: all 0.2s ease;
    }


    div.stButton > button:hover {

        border-color: #2677d9 !important;

        background: #eef6ff !important;

        color: #0756c9 !important;
    }


    /* ---------------------------------------------
       선택된 규정집 버튼
    --------------------------------------------- */

    .selected-button-text {

        background: #eaf4ff;

        border: 1px solid #b9d8f8;

        border-radius: 14px;

        padding: 12px 15px;

        color: #0756c9;

        font-weight: 800;

        margin-bottom: 10px;
    }


    /* ---------------------------------------------
       Streamlit 하단 footer 숨기기
    --------------------------------------------- */

    footer {
        visibility: hidden;
    }


    /* GitHub / 상단 툴바 숨기기 */

    [data-testid="stToolbar"] {
        visibility: hidden;
        height: 0;
    }


    /* ---------------------------------------------
       모바일
    --------------------------------------------- */

    @media (max-width: 600px) {

        .main .block-container {

            padding-left: 16px;
            padding-right: 16px;

            padding-top: 10px;
        }


        .hero {

            border-radius: 0 0 22px 22px;

            padding: 24px 20px 28px 20px;
        }


        .hero-title {

            font-size: 30px;
        }


        .hero-subtitle {

            font-size: 14px;
        }


        .info-card {

            padding: 20px;
        }


        .user-bubble {

            max-width: 90%;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 3. 상단 메인 화면
# =========================================================

st.html(
    """
    <div class="hero">

        <div class="hero-top">

            <div class="hero-logo">
                📚
            </div>

            <div>
                <div>주택관리공단</div>
                <div style="font-size:12px; opacity:0.9;">
                    대구경북지사
                </div>
            </div>

        </div>


        <div class="hero-title">
            기술업무 AI 챗봇
        </div>


        <div class="hero-subtitle">

            기술업무 관련 규정·지침을<br>
            등록된 문서에 근거하여<br>
            쉽고 정확하게 답변해 드립니다.

        </div>

    </div>
    """
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
# 10. 등록된 PDF 목록
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
# 11. 현재 선택된 규정집
# =========================================================

if "selected_file" not in st.session_state:

    st.session_state.selected_file = None


# =========================================================
# 12. 안내 카드
# =========================================================

st.html(
    """
    <div class="info-card">

        <div class="info-title">
            🤖 안녕하십니까.
        </div>

        <div class="info-text">

            저는 <b>대구경북지사 기술업무 담당 AI 챗봇</b>입니다.<br><br>

            등록된 기술업무 규정집을 벡터 검색하여
            질문과 관련성이 높은 내용을 찾아 답변해 드립니다.

        </div>

    </div>
    """
)


# =========================================================
# 13. 선택된 규정집 표시
# =========================================================

if st.session_state.selected_file:

    st.html(
        f"""
        <div class="selected-card">

            <div class="selected-label">
                📚 현재 선택된 규정집
            </div>

            <div class="selected-file">
                {st.session_state.selected_file}
            </div>

        </div>
        """
    )


# =========================================================
# 14. 선택 안내 문구
# =========================================================

st.markdown(
    """
    **※ 선택한 규정집에 명시되지 않은 내용은
    임의로 답변하지 않습니다.**
    """
)


# =========================================================
# 15. 규정집 목록 버튼
# =========================================================

if st.button(
    "📚 규정집 목록",
    use_container_width=True
):

    st.session_state.show_documents = (
        not st.session_state.get(
            "show_documents",
            False
        )
    )


# =========================================================
# 16. 규정집 목록 표시
# =========================================================

if st.session_state.get(
    "show_documents",
    False
):

    st.markdown(
        "### 📚 질문할 규정집을 선택하세요."
    )

    for filename in filenames:

        if st.button(
            f"📋 {filename}",
            key=f"document_{filename}",
            use_container_width=True
        ):

            st.session_state.selected_file = filename

            st.session_state.show_documents = False

            # 규정집 변경 시 대화 초기화
            st.session_state.messages = []

            st.rerun()


# =========================================================
# 17. 선택된 규정집이 없는 경우
# =========================================================

if not st.session_state.selected_file:

    st.html(
        """
        <div class="info-card">

            <div class="info-title">
                📚 규정집을 먼저 선택해주세요.
            </div>

            <div class="info-text">

                위의 <b>📚 규정집 목록</b> 버튼을 눌러
                질문할 규정집을 선택하면 됩니다.

            </div>

        </div>
        """
    )

    st.stop()


# =========================================================
# 18. 검색 함수
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

    # 선택된 문서에 해당하는 결과만 사용하기 위해
    # 먼저 충분히 많이 검색
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


        # -----------------------------------------
        # 선택한 PDF만 검색
        # -----------------------------------------

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
# 19. 대화 기록 초기화
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================================
# 20. 기존 대화 표시
# =========================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        st.html(
            f"""
            <div class="user-bubble">
                {message["content"]}
            </div>
            """
        )

    else:

        st.html(
            f"""
            <div class="ai-card">

                <div class="ai-header">

                    <div class="ai-icon">
                        🤖
                    </div>

                    <div>
                        대구경북 기술업무 AI
                    </div>

                </div>

                <div class="ai-content">

                    {message["content"]}

                </div>

            </div>
            """
        )


# =========================================================
# 21. 질문 입력
# =========================================================

user_input = st.chat_input(
    "선택한 규정집에 대해 질문하세요."
)


# =========================================================
# 22. 질문이 입력된 경우
# =========================================================

if user_input:

    # =====================================================
    # 사용자 질문 저장
    # =====================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )


    # =====================================================
    # 사용자 질문 표시
    # =====================================================

    st.html(
        f"""
        <div class="user-bubble">
            {user_input}
        </div>
        """
    )


    # =====================================================
    # AI 답변
    # =====================================================

    with st.spinner(
        "📚 선택한 규정집에서 관련 내용을 검색하고 있습니다..."
    ):

        try:

            # ---------------------------------------------
            # ① 벡터 검색
            # ---------------------------------------------

            search_results = search_documents(
                user_input,
                st.session_state.selected_file,
                top_k=6
            )


            # ---------------------------------------------
            # ② 관련 규정이 없는 경우
            # ---------------------------------------------

            if not search_results:

                answer = (
                    "해당 내용은 선택한 규정집에 "
                    "명시되어 있지 않습니다."
                )


            else:

                # -----------------------------------------
                # ③ 검색 결과 정리
                # -----------------------------------------

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


                # -----------------------------------------
                # ④ Gemini 프롬프트
                # -----------------------------------------

                prompt = f"""
너는 주택관리공단 대구경북지사의
기술업무 담당 AI 챗봇이다.

사용자가 선택한 규정집의 검색 결과만을
근거로 답변해야 한다.

[선택된 규정집]

{st.session_state.selected_file}

[답변 원칙]

1. 반드시 검색된 규정 문서의 내용을 근거로 답변한다.

2. 검색된 내용에 없는 사항은 추측하지 않는다.

3. 일반적인 지식이나 인터넷 정보를 사용하지 않는다.

4. 답변 근거가 부족하면 다음 문장을 사용한다.

"해당 내용은 선택한 규정집에 명시되어 있지 않습니다."

5. 답변에는 관련 문서명과 페이지를 표시한다.

6. 규정이나 절차는 이해하기 쉽도록
항목별로 정리한다.

7. 검색된 문서에 서로 다른 내용이 있으면
임의로 판단하지 말고 차이를 설명한다.

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


                # -----------------------------------------
                # ⑤ Gemini 호출
                # -----------------------------------------

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
            # 23. AI 답변 표시
            # =================================================

            st.html(
                f"""
                <div class="ai-card">

                    <div class="ai-header">

                        <div class="ai-icon">
                            🤖
                        </div>

                        <div>
                            대구경북 기술업무 AI
                        </div>

                    </div>

                    <div class="ai-content">

                        {answer}

                    </div>

                </div>
                """
            )


            # =================================================
            # 24. 답변 근거 표시
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


                    st.html(
                        f"""
                        <div class="source-card">

                            📄 <b>답변 근거</b><br>

                            📋 {result["filename"]}<br>

                            📖 {result["page"]}페이지

                        </div>
                        """
                    )


            # =================================================
            # 25. AI 답변 저장
            # =================================================

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )


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
