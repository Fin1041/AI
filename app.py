import streamlit as st
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
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
# 3. Gemini 클라이언트
# ==========================================

try:

    client = genai.Client(
        api_key=api_key
    )

except Exception as e:

    st.error(
        f"Gemini 클라이언트 생성 오류: {e}"
    )

    st.stop()


# ==========================================
# 4. 임베딩 모델 불러오기
#
# 무료 로컬 임베딩 모델
# 인터넷 API 호출 없이 서버에서 실행
# ==========================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "intfloat/multilingual-e5-small"
    )

    return model


with st.spinner(
    "🔎 검색 엔진을 준비하고 있습니다..."
):

    embedding_model = load_embedding_model()


# ==========================================
# 5. PDF 파일 찾기
# ==========================================

@st.cache_data
def get_pdf_files():

    pdf_files = glob.glob(
        "**/*.pdf",
        recursive=True
    )

    # 중복 제거
    pdf_files = list(
        dict.fromkeys(pdf_files)
    )

    return pdf_files


pdf_files = get_pdf_files()


# ==========================================
# 6. PDF가 없는 경우
# ==========================================

if not pdf_files:

    st.warning(
        "GitHub 저장소에 PDF 파일(.pdf)을 "
        "하나 이상 업로드해주세요."
    )

    st.stop()


# ==========================================
# 7. PDF 파일 변경 여부 확인용
#
# 파일명 + 수정시간 + 파일크기를 이용
# PDF가 변경되면 자동으로 다시 벡터화
# ==========================================

def get_file_signature(files):

    signature = []

    for file_path in files:

        try:

            stat = os.stat(file_path)

            signature.append(
                (
                    file_path,
                    stat.st_mtime,
                    stat.st_size
                )
            )

        except Exception:
            pass

    return tuple(signature)


file_signature = get_file_signature(
    pdf_files
)


# ==========================================
# 8. PDF → 문서 조각 만들기
# ==========================================

@st.cache_data
def load_pdf_chunks(file_signature):

    chunks = []

    for file_path, _, _ in file_signature:

        filename = os.path.basename(
            file_path
        )

        try:

            reader = PdfReader(
                file_path
            )

            for page_num, page in enumerate(
                reader.pages,
                start=1
            ):

                text = page.extract_text()

                if not text:
                    continue

                # ----------------------------------
                # 불필요한 공백 정리
                # ----------------------------------

                text = text.replace(
                    "\x00",
                    " "
                )

                text = "\n".join(
                    line.strip()
                    for line in text.splitlines()
                    if line.strip()
                )

                if not text:
                    continue


                # ----------------------------------
                # 페이지 하나를 적당한 크기로 분할
                # ----------------------------------

                chunk_size = 1000
                overlap = 150

                start = 0

                while start < len(text):

                    end = start + chunk_size

                    chunk_text = text[
                        start:end
                    ].strip()

                    if chunk_text:

                        chunks.append(
                            {
                                "text": chunk_text,
                                "filename": filename,
                                "page": page_num,
                                "file_path": file_path
                            }
                        )

                    start += (
                        chunk_size - overlap
                    )

        except Exception as e:

            st.error(
                f"'{filename}' 읽기 오류: {e}"
            )

    return chunks


# ==========================================
# 9. PDF 읽기
# ==========================================

with st.spinner(
    "📚 규정집을 분석하고 있습니다..."
):

    document_chunks = load_pdf_chunks(
        file_signature
    )


if not document_chunks:

    st.error(
        "PDF에서 읽을 수 있는 텍스트를 "
        "찾지 못했습니다."
    )

    st.stop()


# ==========================================
# 10. 벡터 DB 생성
#
# PDF 내용을 벡터로 변환
# ==========================================

@st.cache_resource
def create_vector_database(
    chunks,
    signature
):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # --------------------------------------
    # E5 모델은 passage: 를 붙이는 것이 중요
    # --------------------------------------

    passages = [
        "passage: " + text
        for text in texts
    ]

    embeddings = embedding_model.encode(
        passages,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=16
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    # --------------------------------------
    # 코사인 유사도 검색
    # 정규화된 벡터 + Inner Product
    # --------------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


with st.spinner(
    "🧠 규정집을 검색 가능한 형태로 변환하고 있습니다..."
):

    vector_index = create_vector_database(
        document_chunks,
        file_signature
    )


# ==========================================
# 11. 벡터 검색 함수
# ==========================================

def search_documents(
    query,
    top_k=6
):

    # 질문 벡터화
    query_embedding = embedding_model.encode(
        [
            "query: " + query
        ],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # --------------------------------------
    # 가장 관련성이 높은 문서 검색
    # --------------------------------------

    scores, indices = vector_index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index in zip(
        scores[0],
        indices[0]
    ):

        if index < 0:
            continue

        result = document_chunks[
            int(index)
        ].copy()

        result["score"] = float(
            score
        )

        results.append(
            result
        )

    return results


# ==========================================
# 12. 첫 화면 안내 메시지
# ==========================================

if "messages" not in st.session_state:

    welcome_message = """
안녕하십니까.

저는 **대구경북지사 기술업무 담당 AI 챗봇**입니다.

아래 기술업무 관련 문서를 벡터 검색하여
질문과 가장 관련성이 높은 내용을 찾아 답변해 드립니다.

### 📚 현재 등록된 규정집
"""

    for file_path in pdf_files:

        filename = os.path.basename(
            file_path
        )

        welcome_message += (
            f"- 📋 {filename}\n"
        )

    welcome_message += """

궁금하신 사항을 질문해 주시면
**관련 규정과 문서를 찾아 근거를 표시하여 답변**해 드리겠습니다.

※ 업로드된 문서에서 답변 근거를 찾을 수 없는 경우
임의로 답변하지 않습니다.
"""

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": welcome_message
        }
    ]


# ==========================================
# 13. 기존 대화 표시
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ==========================================
# 14. 사용자 질문
# ==========================================

user_input = st.chat_input(
    "규정이나 기술업무에 대해 질문하세요."
)


# ==========================================
# 15. 질문 처리
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

        st.markdown(
            user_input
        )


    # ======================================
    # 16. 벡터 검색
    # ======================================

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 관련 규정을 검색하고 있습니다..."
        ):

            search_results = search_documents(
                user_input,
                top_k=6
            )


            # ==================================
            # 17. 검색 결과가 없는 경우
            # ==================================

            if not search_results:

                answer = (
                    "해당 내용은 업로드된 "
                    "기술업무 문서에 명시되어 있지 않습니다."
                )

                st.markdown(
                    answer
                )

            else:

                # ==================================
                # 18. Gemini에 전달할 검색 결과 만들기
                #
                # PDF 전체가 아니라
                # 관련 내용만 전달
                # ==================================

                context_parts = []

                for i, result in enumerate(
                    search_results,
                    start=1
                ):

                    context_parts.append(
                        f"""
[검색결과 {i}]
문서명: {result["filename"]}
페이지: {result["page"]}페이지
관련도: {result["score"]:.3f}

내용:
{result["text"]}
"""
                    )

                search_context = "\n".join(
                    context_parts
                )


                # ==================================
                # 19. Gemini 프롬프트
                # ==================================

                prompt = f"""
너는 주택관리공단 대구경북지사의
기술업무 담당 AI 챗봇이다.

사용자의 질문에 대해 아래 [검색된 규정 문서]
내용을 근거로 답변해야 한다.

[매우 중요한 답변 원칙]

1. 반드시 아래 검색된 문서 내용을
   근거로 답변한다.

2. 검색된 문서에 없는 내용을
   추측하거나 만들어내지 않는다.

3. 검색된 내용만으로 질문에 대한
   명확한 답변을 할 수 없는 경우에는
   다음 문장을 사용한다.

"해당 내용은 업로드된 기술업무 문서에 명시되어 있지 않습니다."

4. 답변의 근거가 되는 PDF 파일명을 반드시 표시한다.

5. 가능한 경우 페이지 번호를 표시한다.

6. 여러 문서가 관련된 경우
   관련 문서를 모두 표시한다.

7. 인터넷 검색을 사용하지 않는다.

8. 일반적인 지식보다
   제공된 규정 문서를 우선한다.

9. 답변은 이해하기 쉬운 한국어로 작성한다.

10. 규정이나 업무절차를 설명할 때는
    항목별로 정리한다.

11. 서로 다른 규정의 내용이 충돌하는 경우
    임의로 판단하지 말고
    충돌 사실을 알려준다.

━━━━━━━━━━━━━━━━━━━━━━━━━━
[검색된 관련 규정]
━━━━━━━━━━━━━━━━━━━━━━━━━━

{search_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━
[사용자 질문]
━━━━━━━━━━━━━━━━━━━━━━━━━━

{user_input}

━━━━━━━━━━━━━━━━━━━━━━━━━━
[답변]
━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


                # ==================================
                # 20. Gemini 호출
                # ==================================

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

                        # 503 재시도
                        if (
                            "503" in error_text
                            or
                            "UNAVAILABLE"
                            in error_text
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


                # ==================================
                # 21. 최종 답변
                # ==================================

                if not answer:

                    answer = (
                        "⚠️ 현재 Gemini AI 서버가 "
                        "일시적으로 응답하지 않습니다.\n\n"
                        "잠시 후 다시 질문해 주세요."
                    )


                # ==================================
                # 22. 답변 표시
                # ==================================

                st.markdown(
                    answer
                )


                # ==================================
                # 23. 검색된 출처 표시
                # ==================================

                st.markdown(
                    "---"
                )

                st.markdown(
                    "### 📚 답변 근거 문서"
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
                        f"📋 **{result['filename']}**  "
                        f"— {result['page']}페이지"
                    )


    # ==========================================
    # 24. 답변 저장
    # ==========================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
