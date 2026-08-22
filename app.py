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


# ==================================================
# 2. Streamlit 기본 UI 숨기기
# ==================================================

st.markdown("""
<style>

/* 오른쪽 위 도구 모음 */
[data-testid="stToolbar"] {
    display: none !important;
}

/* 상단 장식 */
[data-testid="stDecoration"] {
    display: none !important;
}

/* 하단 기본 footer */
footer {
    display: none !important;
}

/* Streamlit 상태 위젯 */
[data-testid="stStatusWidget"] {
    display: none !important;
}

/* Streamlit 배지 */
[class*="viewerBadge"] {
    display: none !important;
}

[class*="stAppDeployButton"] {
    display: none !important;
}


/* ------------------------------------------
   PDF 선택 버튼
------------------------------------------ */

div.stButton > button {
    width: 100%;
    min-height: 58px;
    text-align: left;
    font-size: 16px;
    font-weight: 500;
    border-radius: 12px;
}


/* ------------------------------------------
   모바일 화면 여백
------------------------------------------ */

.block-container {
    padding-top: 2rem;
    padding-bottom: 5rem;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. 제목
# ==================================================

st.title(
    "📚 대구경북 기술업무 AI 챗봇"
)


# ==================================================
# 4. Gemini API 확인
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
# 5. Gemini 클라이언트
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
# 6. 임베딩 모델
# ==================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "intfloat/multilingual-e5-small"
    )


with st.spinner(
    "🔎 검색 시스템을 준비하고 있습니다..."
):

    embedding_model = load_embedding_model()


# ==================================================
# 7. 벡터 DB 위치
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
# 8. 벡터 DB 확인
# ==================================================

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


# ==================================================
# 9. 벡터 DB 불러오기
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

        documents = pickle.load(f)

    return index, documents


with st.spinner(
    "📚 규정집 검색 DB를 불러오는 중..."
):

    vector_index, documents = (
        load_vector_database()
    )


# ==================================================
# 10. 등록된 PDF 파일 목록 만들기
# ==================================================

filenames = []

for document in documents:

    filename = str(
        document.get("filename") or "파일명 없음"
    )

    if filename not in filenames:

        filenames.append(filename)


# ==================================================
# 11. 선택된 PDF 상태
# ==================================================

if "selected_file" not in st.session_state:

    st.session_state.selected_file = None


if "messages" not in st.session_state:

    st.session_state.messages = []


# ==================================================
# 12. PDF 선택 화면
# ==================================================

if st.session_state.selected_file is None:

    st.markdown(
        """
안녕하십니까.

저는 **대구경북지사 기술업무 담당 AI 챗봇**입니다.

등록된 기술업무 규정집을 벡터 검색하여
질문과 관련성이 높은 규정을 찾아 답변해 드립니다.

### 📚 질문할 규정집을 선택하세요.
"""
    )


    if not filenames:

        st.warning(
            "등록된 규정집이 없습니다."
        )

        st.stop()


    # ----------------------------------------------
    # PDF 선택 버튼
    # ----------------------------------------------

    for filename in filenames:

        if st.button(
            f"📋  {filename}",
            key=f"pdf_{filename}"
        ):

            st.session_state.selected_file = (
                filename
            )

            # 이전 대화 삭제
            st.session_state.messages = []

            st.rerun()


    st.markdown(
        """
<br>

※ 규정집을 선택하면 선택한 문서에 대해서만
질문하고 답변받을 수 있습니다.
""",
        unsafe_allow_html=True
    )


    st.stop()




# ==================================================
# 16. 첫 질문 전 안내
# ==================================================

if not st.session_state.messages:

    welcome = f"""
안녕하십니까.

현재 **「{selected_file}」** 규정집을 선택하셨습니다.

이제 이 규정집에 대해서만 질문할 수 있습니다.

예를 들어,

- 해당 업무의 처리절차는 무엇인가요?
- 담당자의 업무는 무엇인가요?
- 적용 대상은 어떻게 되나요?
- 관련 기준은 무엇인가요?

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


# ==================================================
# 17. 기존 대화 표시
# ==================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ==================================================
# 14. 규정집 목록으로 돌아가기
# ==================================================

if st.button(
    "← 규정집 목록",
    key="back_to_list"
):

    st.session_state.selected_file = None

    st.session_state.messages = []

    st.rerun()
    
# ==================================================
# 18. 벡터 검색 함수
# ==================================================

def search_selected_document(
    query,
    selected_filename,
    top_k=6
):

    # ----------------------------------------------
    # 질문 확인
    # ----------------------------------------------

    if query is None:

        return []


    query = str(
        query
    ).strip()


    if not query:

        return []


    # ----------------------------------------------
    # 질문 벡터화
    # ----------------------------------------------

    query_embedding = (
        embedding_model.encode(
            ["query: " + query],
            normalize_embeddings=True
        )
    )


    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )


    # ----------------------------------------------
    # 전체 벡터에서 충분히 많이 검색
    #
    # 선택한 PDF의 결과만 나중에 필터링
    # ----------------------------------------------

    total_vectors = vector_index.ntotal


    if total_vectors <= 0:

        return []


    search_count = min(
        max(top_k * 20, 50),
        total_vectors
    )


    scores, indices = (
        vector_index.search(
            query_embedding,
            search_count
        )
    )


    results = []


    # ----------------------------------------------
    # 선택한 PDF만 남김
    # ----------------------------------------------

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:

            continue


        idx = int(idx)


        if idx >= len(documents):

            continue


        result = documents[idx]


        filename = str(
            result.get("filename") or ""
        )


        # ==========================================
        # 핵심
        #
        # 현재 선택한 PDF가 아니면 제외
        # ==========================================

        if filename != selected_filename:

            continue


        text = str(
            result.get("text") or ""
        ).strip()


        if not text:

            continue


        result_copy = result.copy()


        result_copy["filename"] = (
            filename
        )


        result_copy["page"] = str(
            result.get("page")
            or "페이지 정보 없음"
        )


        result_copy["text"] = text


        result_copy["score"] = float(
            score
        )


        results.append(
            result_copy
        )


        # 필요한 개수만 확보
        if len(results) >= top_k:

            break


    return results


# ==================================================
# 19. 사용자 질문
# ==================================================

user_input = st.chat_input(
    "규정이나 기술업무에 대해 질문하세요."
)


# ==================================================
# 20. 질문했을 때만 실행
#
# ★ 매우 중요
#
# 기존 코드의 오류를 수정하여
# user_input이 있을 때만 AI가 실행됩니다.
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


    # ----------------------------------------------
    # AI 답변
    # ----------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        status = st.status(
            "📚 관련 규정을 검색하고 있습니다...",
            expanded=True
        )


        try:

            # ======================================
            # ① 선택된 PDF만 벡터 검색
            # ======================================

            search_results = (
                search_selected_document(
                    user_input,
                    selected_file,
                    top_k=6
                )
            )


            # ======================================
            # ② 검색 결과 없음
            # ======================================

            if not search_results:

                status.update(
                    label="❌ 선택한 규정집에서 관련 내용을 찾지 못했습니다.",
                    state="complete",
                    expanded=False
                )


                answer = (
                    "해당 내용은 선택하신 규정집에 "
                    "명시되어 있지 않습니다."
                )


                st.markdown(
                    answer
                )


            else:

                # ==================================
                # 검색 결과 확인
                # ==================================

                status.update(
                    label=(
                        f"✅ 「{selected_file}」에서 "
                        f"관련 규정 {len(search_results)}건을 찾았습니다."
                    ),
                    state="running",
                    expanded=True
                )


                # ==================================
                # 검색 결과를 Gemini에 전달
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


                search_context = (
                    "\n".join(
                        context_parts
                    )
                )


                # ==================================
                # Gemini 프롬프트
                # ==================================

                prompt = f"""
너는 주택관리공단 대구경북지사의
기술업무 담당 AI 챗봇이다.

사용자가 현재 선택한 규정집은 다음과 같다.

[선택된 규정집]

{selected_file}


매우 중요하다.

사용자의 질문에 대해 반드시
위에서 선택된 규정집의
[검색된 관련 내용]만을 근거로 답변해야 한다.

다른 PDF 문서의 내용이나
인터넷 검색 결과를 사용하지 않는다.


[답변 원칙]

1. 선택된 규정집의 검색 결과만 근거로 답변한다.

2. 검색 결과에 없는 내용을
추측하거나 만들어내지 않는다.

3. 답변 근거가 부족한 경우에는
다음과 같이 답변한다.

"해당 내용은 선택하신 규정집에 명시되어 있지 않습니다."

4. 답변에는 가능한 경우
관련 페이지를 표시한다.

5. 규정이나 업무절차는
이해하기 쉽도록 항목별로 정리한다.

6. 검색된 규정 내용에 서로 다른 내용이 있으면
임의로 하나를 선택하지 말고
차이가 있음을 설명한다.

7. 일반적인 지식보다
검색된 규정 내용을 우선한다.

8. 인터넷 검색을 하지 않는다.

9. 답변은 한국어로 작성한다.

10. 질문과 직접 관련이 없는 내용은
답변에 포함하지 않는다.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[검색된 관련 내용]

{search_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[사용자 질문]

{user_input}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[답변]

"""


                # ==================================
                # Gemini 답변
                # ==================================

                status.update(
                    label=(
                        "🤖 관련 규정을 확인했습니다. "
                        "답변을 작성하고 있습니다..."
                    ),
                    state="running",
                    expanded=True
                )


                answer = None


                # 최대 3회 재시도
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


                        # 503 재시도
                        if (
                            "503"
                            in error_text
                            or
                            "UNAVAILABLE"
                            in error_text
                        ):

                            if attempt < 2:

                                status.update(
                                    label=(
                                        f"🔄 AI 서버 재시도 중... "
                                        f"({attempt + 1}/3)"
                                    ),
                                    state="running",
                                    expanded=True
                                )


                                time.sleep(
                                    2 ** attempt
                                )


                                continue


                        # 기타 오류
                        answer = (
                            "⚠️ Gemini AI 오류가 발생했습니다.\n\n"
                            f"`{error_text}`"
                        )


                        break


                # ==================================
                # 답변 실패
                # ==================================

                if not answer:

                    answer = (
                        "⚠️ 현재 Gemini AI 서버가 "
                        "일시적으로 응답하지 않습니다.\n\n"
                        "잠시 후 다시 질문해 주세요."
                    )


                # ==================================
                # 완료
                # ==================================

                status.update(
                    label="✅ 답변 작성이 완료되었습니다.",
                    state="complete",
                    expanded=False
                )


                # ==================================
                # 답변 표시
                # ==================================

                st.markdown(
                    answer
                )


                # ==================================
                # 답변 근거는 화면에 별도로 표시하지 않음
                #
                # 사용자가 원하셨던
                # "답변 근거" 목록을 제거
                # ==================================


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


            st.error(
                answer
            )


    # ----------------------------------------------
    # 답변 저장
    # ----------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
