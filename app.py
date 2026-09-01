import streamlit as st
from google import genai

import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer

import os
import time
import zipfile
import tempfile
import re
import copy
import xml.etree.ElementTree as ET
import html
import urllib.request
import json
import io


# =========================================================
# 1. 페이지 설정
# =========================================================

st.set_page_config(
    page_title="주택관리공단 대구경북지사",
    page_icon=" ",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# =========================================================
# 2. 화면 디자인
#    ※ 기존 검색/AI 기능은 변경하지 않고
#       화면 디자인만 AI 비서형으로 변경
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


    /* 가운데 전체 영역 */

    .main .block-container {
        max-width: 720px;
        padding-top: 18px;
        padding-left: 18px;
        padding-right: 18px;
        padding-bottom: 80px;
    }


    /* Streamlit 기본 요소 */

    footer {
        visibility: hidden;
    }

    [data-testid="stToolbar"] {
        visibility: hidden;
        height: 0;
    }


    /* =====================================================
       상단 제목
       ===================================================== */

    .top-company {
        font-size: 18px;
        font-weight: 700;
        color: #20364f;
        margin-top: 5px;
        margin-bottom: 3px;
    }

    .top-title {
        font-size: 30px;
        font-weight: 800;
        color: #16283d;
        margin-top: 0px;
        margin-bottom: 5px;
        letter-spacing: -1.5px;
    }

    .top-subtitle {
        font-size: 14px;
        color: #718096;
        margin-bottom: 15px;
    }


    /* =====================================================
       AI 로봇 영역
       ===================================================== */

    .ai-avatar {
        text-align: center;
        font-size: 70px;
        line-height: 1;
        margin-top: 20px;
        margin-bottom: 8px;
    }


    .ai-greeting {
        text-align: center;
        color: #172b43;
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-top: 8px;
        margin-bottom: 5px;
    }


    .ai-description {
        text-align: center;
        color: #7a8797;
        font-size: 14px;
        margin-bottom: 22px;
    }


    /* =====================================================
       안내 카드
       ===================================================== */

    .welcome-card {
        background: #ffffff;
        border: 1px solid #e3ebf5;
        border-radius: 22px;
        padding: 18px 20px;
        margin-top: 8px;
        margin-bottom: 18px;
        box-shadow: 0 5px 20px rgba(35, 80, 130, 0.06);
    }


    .welcome-title {
        color: #175ca8;
        font-size: 15px;
        font-weight: 800;
        margin-bottom: 8px;
    }


    .welcome-text {
        color: #66758a;
        font-size: 13px;
        line-height: 1.7;
    }


    /* =====================================================
       추천 질문 제목
       ===================================================== */

    .section-title {
        color: #24364d;
        font-size: 17px;
        font-weight: 800;
        margin-top: 18px;
        margin-bottom: 10px;
    }


    /* =====================================================
       버튼 디자인
       ===================================================== */

    div.stButton > button {
        width: 100%;
        min-height: 48px !important;

        border-radius: 15px !important;

        border: 1px solid #dce7f4 !important;

        background-color: #ffffff !important;

        color: #29435f !important;

        font-size: 14px !important;

        font-weight: 600 !important;

        box-shadow: 0 3px 12px rgba(35, 80, 130, 0.04);

        transition: 0.2s;
    }


    div.stButton > button:hover {
        background-color: #f1f7ff !important;

        border-color: #8dbcf0 !important;

        color: #1765b5 !important;

        transform: translateY(-1px);
    }


    /* =====================================================
       규정집 버튼
       ===================================================== */

    .document-area {
        margin-top: 25px;
    }


    /* =====================================================
       선택된 규정집
       ===================================================== */

    .selected-document {
        background: #eaf4ff;
        border: 1px solid #c9e1fa;
        border-radius: 16px;
        padding: 14px 16px;
        margin-top: 8px;
        margin-bottom: 12px;
    }


    .selected-document-title {
        color: #1761a9;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 4px;
    }


    .selected-document-name {
        color: #243c56;
        font-size: 14px;
        font-weight: 700;
        line-height: 1.5;
        word-break: keep-all;
    }


    /* =====================================================
       채팅 영역
       ===================================================== */

    [data-testid="stChatMessage"] {
        border-radius: 18px;
        margin-bottom: 10px;
    }


    /* 사용자 메시지 */

    [data-testid="stChatMessage"]:has(
        [data-testid="chatAvatarIcon-user"]
    ) {
        background-color: #eaf4ff;
    }


    /* =====================================================
       Chat Input
       ===================================================== */

    [data-testid="stChatInput"] {
        padding-bottom: 10px;
    }


    [data-testid="stChatInput"] textarea {
        border-radius: 18px !important;
        border: 1px solid #cbdff5 !important;
        background: white !important;
        min-height: 52px !important;
        font-size: 14px !important;
    }


    [data-testid="stChatInput"] textarea:focus {
        border-color: #4e93db !important;
        box-shadow: 0 0 0 2px rgba(78, 147, 219, 0.12) !important;
    }


    /* =====================================================
       구분선
       ===================================================== */

    hr {
        border: none;
        border-top: 1px solid #dce5ef;
        margin-top: 18px;
        margin-bottom: 18px;
    }


    /* =====================================================
       모바일
       ===================================================== */

    @media (max-width: 600px) {

        .main .block-container {
            padding-left: 14px;
            padding-right: 14px;
            padding-top: 10px;
            padding-bottom: 70px;
        }


        .top-company {
            font-size: 16px;
        }


        .top-title {
            font-size: 27px;
        }


        .ai-avatar {
            font-size: 62px;
            margin-top: 15px;
        }


        .ai-greeting {
            font-size: 22px;
        }


        .welcome-card {
            border-radius: 19px;
            padding: 16px;
        }


        div.stButton > button {
            min-height: 46px !important;
            font-size: 13px !important;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 3. 세션 상태
# =========================================================

if "selected_file" not in st.session_state:
    st.session_state.selected_file = None


if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_notice_generator" not in st.session_state:
    st.session_state.show_notice_generator = False


# =========================================================
# 4. 상단 화면
# =========================================================



st.markdown(
    '<div class="top-title">🏠 대구경북지사 기술업무 AI 챗봇 🤖</div>',
    unsafe_allow_html=True
)


st.markdown("---")


# =========================================================
# 5. Gemini API 확인
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
# 6. Gemini 클라이언트
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
# 7. 임베딩 모델
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
# 8. 벡터 DB
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
# 9. 벡터 DB 확인
# =========================================================

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
# 10. 벡터 DB 불러오기
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
# 11. PDF 목록 만들기
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


    # ---------------------------------------------
    # 질문을 벡터로 변환
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


        # -----------------------------------------
        # 선택한 PDF만 사용
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




# ---------------------------------------------------------
# GitHub 설정
# ---------------------------------------------------------

GITHUB_USERNAME = "Fin1041"
GITHUB_REPOSITORY = "AI"
GITHUB_BRANCH = "main"

GITHUB_TOKEN = ""

PLAN_PDF_PATH = "templates/plan.pdf"
HWPX_TEMPLATE_PATH = "templates/notice_template.hwpx"


def _github_raw_url(path):
    return (
        "https://raw.githubusercontent.com/"
        f"{GITHUB_USERNAME}/"
        f"{GITHUB_REPOSITORY}/"
        f"{GITHUB_BRANCH}/"
        f"{path}"
    )


HWPX_TEMPLATE_URLS = [
    _github_raw_url(HWPX_TEMPLATE_PATH)
]


def _github_headers():
    headers = {
        "User-Agent": "house-management-notice-app"
    }

    if GITHUB_TOKEN:
        headers["Authorization"] = (
            f"Bearer {GITHUB_TOKEN}"
        )

    return headers


@st.cache_data(ttl=3600)
def _get_plan_file_urls():
    """GitHub의 templates/plan.pdf 단일 규정집 URL."""
    return [
        {
            "name": "plan.pdf",
            "url": _github_raw_url(PLAN_PDF_PATH)
        }
    ]


def _load_plan_regulation_pages():
    """
    templates/plan.pdf를 페이지별 텍스트로 읽는다.
    """
    files = _get_plan_file_urls()

    pages = []

    if not files:
        return pages

    try:
        from pypdf import PdfReader
    except Exception:
        return pages

    for file_info in files:

        request = urllib.request.Request(
            file_info["url"],
            headers=_github_headers()
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=30
            ) as response:
                pdf_bytes = response.read()
        except Exception:
            continue

        try:
            reader = PdfReader(
                io.BytesIO(pdf_bytes)
            )
        except Exception:
            continue

        for page_no, page in enumerate(
            reader.pages,
            start=1
        ):

            try:
                text = (
                    page.extract_text()
                    or ""
                ).strip()
            except Exception:
                continue

            if not text:
                continue

            pages.append(
                {
                    "filename": file_info["name"],
                    "page": page_no,
                    "text": text
                }
            )

    return pages


@st.cache_resource
def _load_plan_embeddings():
    """
    PLAN 규정집 페이지를 임베딩하여 캐시한다.
    """
    pages = _load_plan_regulation_pages()

    if not pages:
        return None, []

    texts = [
        "passage: " + item["text"][:5000]
        for item in pages
    ]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    return embeddings, pages


def search_plan_regulations(
    query,
    top_k=5,
    min_score=0.25
):
    """
    templates/plan 규정집을 우선 검색한다.
    관련 근거가 충분한 경우만 반환한다.
    """

    query = str(
        query or ""
    ).strip()

    if not query:
        return []

    embeddings, pages = (
        _load_plan_embeddings()
    )

    if embeddings is None or not pages:
        return []

    query_embedding = embedding_model.encode(
        ["query: " + query],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores = (
        embeddings @ query_embedding[0]
    )

    order = np.argsort(
        scores
    )[::-1]

    results = []

    # 같은 규정집 페이지가 지나치게 반복되지 않도록 제한
    per_file = {}

    for idx in order:

        score = float(
            scores[idx]
        )

        if score < min_score:
            continue

        item = pages[int(idx)]
        filename = item["filename"]

        per_file.setdefault(
            filename,
            0
        )

        if per_file[filename] >= 2:
            continue

        result = dict(item)
        result["score"] = score

        results.append(
            result
        )

        per_file[filename] += 1

        if len(results) >= top_k:
            break

    return results




# =========================================================
# AI 안내문 생성
# =========================================================

def download_hwpx_template(urls):
    """
    GitHub에서 HWPX 원본 양식을 다운로드한다.
    새 템플릿(번호형 안내내용)을 먼저 시도하고
    실패하면 기존 notice_template.hwpx를 시도한다.
    """

    if isinstance(urls, str):
        urls = [urls]

    errors = []

    for url in urls:

        try:

            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"}
            )

            with urllib.request.urlopen(
                request,
                timeout=30
            ) as response:
                data = response.read()

            if not data:
                raise RuntimeError("빈 파일입니다.")

            with zipfile.ZipFile(
                __import__("io").BytesIO(data),
                "r"
            ) as z:

                names = z.namelist()

                if not names or names[0] != "mimetype":
                    raise RuntimeError(
                        "정상적인 HWPX ZIP이 아닙니다."
                    )

                if z.read("mimetype") != b"application/hwp+zip":
                    raise RuntimeError(
                        "HWPX mimetype이 올바르지 않습니다."
                    )

                if "Contents/section0.xml" not in names:
                    raise RuntimeError(
                        "section0.xml이 없습니다."
                    )

                section = z.read(
                    "Contents/section0.xml"
                ).decode("utf-8")

                # 새 양식 우선 확인
                has_new = all(
                    f"{{{{안내내용{i}}}}}" in section
                    for i in range(1, 6)
                )

                # 구 양식도 허용
                has_old = "{{안내내용}}" in section

                if not (has_new or has_old):
                    raise RuntimeError(
                        "안내내용 placeholder를 찾지 못했습니다."
                    )

            f = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".hwpx"
            )

            try:
                f.write(data)
                f.flush()
            finally:
                f.close()

            return f.name

        except Exception as e:
            errors.append(
                f"{url}: {e}"
            )

    raise RuntimeError(
        "GitHub에서 정상적인 안내문 템플릿을 찾지 못했습니다.\n\n"
        + "\n".join(errors)
        + "\n\n"
        "GitHub templates 폴더에 "
        "`notice_template.hwpx`를 올려주세요."
    )

def _gemini_notice_request(prompt):
    """Gemini 일시 오류를 최대 5회 재시도."""
    last_error = None

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt
            )

            text = (response.text or "").strip()

            if text:
                return text

            raise RuntimeError("Gemini가 빈 응답을 반환했습니다.")

        except Exception as e:
            last_error = e
            err = str(e).upper()

            temporary = (
                "503" in err
                or "UNAVAILABLE" in err
                or "429" in err
                or "RESOURCE_EXHAUSTED" in err
            )

            if not temporary:
                raise

            if attempt < 4:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        "Gemini 서버가 일시적으로 혼잡합니다. "
        "잠시 후 다시 시도해주세요.\n"
        f"마지막 오류: {last_error}"
    )





def generate_notice_text(
    request_text,
    subject,
    date_value,
    company,
    phone,
    office,
    regulation_context=""
):
    """
    제목 15자 이하 / 본문 최대 5줄.
    본문은 20~25글자 정도를 목표로 짧게 작성한다.
    규정 근거가 확인될 경우 1줄 정도로 간단히 반영한다.
    """

    prompt = f"""
너는 공동주택 관리사무소의 공식 안내문 작성 담당자이다.

[건명]
{subject}

[사용자 요청]
{request_text}

[관련 규정집 검색 결과]
{regulation_context if regulation_context else "관련 규정이 검색되지 않았습니다."}

[작성 규칙]
1. 제목은 반드시 15자 이내다.
2. 제목에는 건명의 핵심어를 포함한다.
3. 안내내용은 반드시 5줄 이내다.
4. 가능하면 4~5줄로 작성한다.
5. 한 줄은 공백 제외 약 20~25글자를 목표로 한다.
6. 각 문장은 반드시 한 줄씩 작성한다.
7. 내용은 건명에 대한 목적, 관리 필요성, 입주민 협조사항 중심으로 간단히 작성한다.
8. templates/plan.pdf에서 실제 규정이나 기준이 확인되면 그 명칭 또는 핵심 내용을 짧게 1줄 포함한다.
9. 검색 결과에 없는 법조문, 의무사항, 과태료, 처벌을 절대 만들어내지 않는다.
10. 일시, 날짜, 업체명, 전화번호, 관리소명은 안내내용에 넣지 않는다.
11. 같은 내용을 반복하지 않는다.
12. 어려운 법률 표현을 줄이고 입주민이 바로 이해할 수 있게 작성한다.
13. 본문은 5줄을 넘기지 않는다.

[출력 형식]
[제목]
15자 이내 제목

[본문]
문장1
문장2
문장3
문장4
문장5

[근거]
확인된 규정/기준이 있으면 짧게 1줄
"""

    text = None
    last_error = None

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt
            )
            text = (response.text or "").strip()
            if text:
                break
            last_error = "Gemini 빈 응답"
        except Exception as e:
            last_error = e
            upper = str(e).upper()
            if not (
                "503" in upper
                or "UNAVAILABLE" in upper
                or "429" in upper
                or "RESOURCE_EXHAUSTED" in upper
            ):
                raise
            if attempt < 4:
                time.sleep(2 ** attempt)

    if not text:
        raise RuntimeError(
            "Gemini 서버가 현재 응답하지 않습니다. "
            f"마지막 오류: {last_error}"
        )

    title = ""
    body = ""
    basis = ""

    if "[제목]" in text:
        rest = text.split("[제목]", 1)[1]
        if "[본문]" in rest:
            title, rest = rest.split("[본문]", 1)
            if "[근거]" in rest:
                body, basis = rest.split("[근거]", 1)
            else:
                body = rest
        else:
            title = rest

    title = re.sub(r"\s+", " ", title.strip())[:15]
    if not title:
        title = (subject.strip() or "안내문")[:15]

    body = re.sub(
        r"^\s*(?:[-•·]|\d+[\.\)])\s*",
        "",
        body,
        flags=re.MULTILINE
    )

    lines = [
        line.strip()
        for line in body.splitlines()
        if line.strip()
    ]

    if len(lines) == 1:
        one = lines[0]
        one = re.sub(
            r"(?<=[다요함됨임])\.\s+",
            ".\n",
            one
        )
        lines = [
            x.strip()
            for x in one.splitlines()
            if x.strip()
        ]

    forbidden = [
        str(v).strip()
        for v in [date_value, company, phone, office]
        if str(v).strip()
    ]

    cleaned = [
        line
        for line in lines
        if not any(f in line for f in forbidden)
    ]

    # 본문은 무조건 5줄 이내
    body = "\n".join(cleaned[:5])

    return title, body, basis

def generate_notice_content(
    request_text,
    subject,
    date_value,
    company,
    phone,
    office,
    regulation_context=""
):
    """
    제목 최대 15자 / 안내내용 최대 5문장.
    본문에는 일시·업체·전화번호·관리소명을 넣지 않는다.
    """

    prompt = f"""
너는 공동주택 관리사무소의 공식 안내문 작성 담당자이다.

[건명]
{subject}

[사용자 요청]
{request_text}

[관련 규정집 검색 결과]
{regulation_context if regulation_context else "관련 규정집에서 관련 내용을 찾지 못했습니다."}

[참고 정보]
일시: {date_value}
업체: {company}
전화번호: {phone}
관리소명: {office}

[작성 규칙]
1. 제목은 반드시 15자 이내로 작성한다.
2. 제목은 건명의 핵심을 포함하여 짧고 명확하게 작성한다.
3. 안내내용은 최대 5문장으로 작성한다.
4. 각 문장은 반드시 한 줄씩 구분한다.
5. 건명과 관련된 작업 목적, 관리 필요성, 입주민 협조사항을 중심으로 작성한다.
6. 등록된 규정집 검색 결과에 규정 또는 기준이 있으면 그것을 최우선 근거로 사용한다.
7. 검색 결과에 없는 법조문 번호, 법적 의무사항, 과태료·처벌 등을 만들어내지 않는다.
8. 일시, 날짜, 업체명, 전화번호, 관리소명은 안내내용에서 절대로 언급하지 않는다.
9. 확인되지 않은 내용은 사실처럼 단정하지 않는다.
10. 정중하고 간결한 행정문체를 사용한다.
11. 불필요한 인사말과 장황한 설명은 생략한다.

[출력 형식]
[제목]
제목

[본문]
문장 1
문장 2
문장 3
문장 4
문장 5

[근거]
확인된 규정/기준만 간단히 작성한다.
"""

    # Gemini 호출은 최대 5회 재시도
    text = None
    last_error = None

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt
            )
            text = (response.text or "").strip()

            if text:
                break

            last_error = "빈 응답"

        except Exception as e:
            last_error = e
            err = str(e).upper()

            temporary = (
                "503" in err
                or "UNAVAILABLE" in err
                or "429" in err
                or "RESOURCE_EXHAUSTED" in err
            )

            if not temporary:
                raise

            if attempt < 4:
                time.sleep(2 ** attempt)

    if not text:
        raise RuntimeError(
            "Gemini 서버가 현재 응답하지 않습니다. "
            "잠시 후 다시 시도해주세요. "
            f"마지막 오류: {last_error}"
        )

    title = ""
    body = ""
    basis = ""

    if "[제목]" in text:
        rest = text.split("[제목]", 1)[1]

        if "[본문]" in rest:
            title, rest = rest.split("[본문]", 1)

            if "[근거]" in rest:
                body, basis = rest.split("[근거]", 1)
            else:
                body = rest
        else:
            title = rest

    # 제목 15자 제한
    title = re.sub(
        r"\s+",
        " ",
        title.strip()
    )[:15]

    if not title:
        title = (subject.strip() or "안내문")[:15]

    # 본문 줄 정리
    body = re.sub(
        r"^\s*(?:[-•·]|\d+[\.\)])\s*",
        "",
        body,
        flags=re.MULTILINE
    )

    lines = [
        line.strip()
        for line in body.splitlines()
        if line.strip()
    ]

    # 한 줄로 반환했으면 문장 종결 뒤에서 분리
    if len(lines) == 1:
        one = lines[0]
        one = re.sub(
            r"([다요함됨임])\.\s+",
            r"\1.\n",
            one
        )
        lines = [
            line.strip()
            for line in one.splitlines()
            if line.strip()
        ]

    # 입력값이 본문에 직접 포함된 문장은 제거
    forbidden = [
        str(v).strip()
        for v in [date_value, company, phone, office]
        if str(v).strip()
    ]

    cleaned = []
    for line in lines:
        if any(v in line for v in forbidden):
            continue
        cleaned.append(line)

    body = "\n".join(cleaned[:5])

    return title, body, basis




def create_notice_hwpx(
    template_path,
    output_path,
    title,
    body,
    notice_date,
    notice_deadline,
    subject,
    work_date,
    company,
    phone,
    office
):
    """
    notice_template.hwpx의
    {{안내내용1}}~{{안내내용5}} 기존 문단을 사용한다.
    """

    lines = [
        line.strip()
        for line in str(body).splitlines()
        if line.strip()
    ][:5]

    with zipfile.ZipFile(
        template_path,
        "r"
    ) as zin:

        names = zin.namelist()

        if not names or names[0] != "mimetype":
            raise RuntimeError(
                "HWPX 원본 구조가 올바르지 않습니다."
            )

        if "Contents/section0.xml" not in names:
            raise RuntimeError(
                "Contents/section0.xml이 없습니다."
            )

        data = {
            name: zin.read(name)
            for name in names
        }

        section = "Contents/section0.xml"
        xml = data[section].decode("utf-8")

        replacements = {
            "{{공고일자}}": notice_date,
            "{{공고기한}}": notice_deadline,
            "{{제목}}": str(title)[:15],
            "{{건 명}}": subject,
            "{{건명}}": subject,
            "{{일 시}}": work_date,
            "{{일시}}": work_date,
            "{{업 체}}": company,
            "{{업체}}": company,
            "{{전화번호}}": phone,
            "{{관리소명}}": office,
        }

        for key, value in replacements.items():
            xml = xml.replace(
                key,
                html.escape(
                    str(value),
                    quote=False
                )
            )

        # 새 템플릿: 안내내용1~5
        numbered_keys = [
            f"{{{{안내내용{i}}}}}"
            for i in range(1, 6)
        ]

        if all(
            key in xml
            for key in numbered_keys
        ):

            for i, key in enumerate(
                numbered_keys
            ):

                p = re.search(
                    r"<hp:p\b[^>]*>"
                    r"(?:(?!<hp:p\b)[\s\S])*?"
                    + re.escape(key)
                    + r"(?:(?!<hp:p\b)[\s\S])*?"
                    r"</hp:p>",
                    xml,
                    re.DOTALL
                )

                if not p:
                    raise RuntimeError(
                        f"{key} 문단을 찾지 못했습니다."
                    )

                paragraph = p.group(0)

                t = re.search(
                    r"<hp:t\b[^>]*>.*?"
                    + re.escape(key)
                    + r".*?</hp:t>",
                    paragraph,
                    re.DOTALL
                )

                if not t:
                    raise RuntimeError(
                        f"{key}의 텍스트 영역을 찾지 못했습니다."
                    )

                value = (
                    lines[i]
                    if i < len(lines)
                    else ""
                )

                paragraph = (
                    paragraph[:t.start()]
                    + "<hp:t>"
                    + html.escape(
                        value,
                        quote=False
                    )
                    + "</hp:t>"
                    + paragraph[t.end():]
                )

                xml = (
                    xml[:p.start()]
                    + paragraph
                    + xml[p.end():]
                )

        # 구 템플릿: {{안내내용}} + 기존 문단
        elif "{{안내내용}}" in xml:

            p_matches = list(
                re.finditer(
                    r"<hp:p\b[^>]*>.*?</hp:p>",
                    xml,
                    re.DOTALL
                )
            )

            body_idx = next(
                (
                    i
                    for i, match in enumerate(p_matches)
                    if "{{안내내용}}" in match.group(0)
                ),
                None
            )

            if body_idx is None:
                raise RuntimeError(
                    "기존 {{안내내용}} 문단을 찾지 못했습니다."
                )

            targets = p_matches[
                body_idx:body_idx + 5
            ]

            if len(targets) < len(lines):
                raise RuntimeError(
                    "기존 템플릿의 본문 문단이 부족합니다."
                )

            for i in range(
                len(lines) - 1,
                -1,
                -1
            ):

                match = targets[i]
                paragraph = match.group(0)

                t = re.search(
                    r"<hp:t\b[^>]*>.*?</hp:t>",
                    paragraph,
                    re.DOTALL
                )

                if not t:
                    raise RuntimeError(
                        f"본문 {i+1}번째 텍스트 영역을 찾지 못했습니다."
                    )

                paragraph = (
                    paragraph[:t.start()]
                    + "<hp:t>"
                    + html.escape(
                        lines[i],
                        quote=False
                    )
                    + "</hp:t>"
                    + paragraph[t.end():]
                )

                xml = (
                    xml[:match.start()]
                    + paragraph
                    + xml[match.end():]
                )

        else:
            raise RuntimeError(
                "HWPX 템플릿에서 안내내용 위치를 찾지 못했습니다."
            )

        ET.fromstring(xml)
        data[section] = xml.encode("utf-8")

        # ZIP 원본 속성 유지
        with zipfile.ZipFile(
            output_path,
            "w"
        ) as zout:

            for name in names:

                info = copy.copy(
                    zin.getinfo(name)
                )

                if name == "mimetype":
                    info.compress_type = zipfile.ZIP_STORED

                zout.writestr(
                    info,
                    data[name]
                )

    with zipfile.ZipFile(
        output_path,
        "r"
    ) as check:

        if check.namelist()[0] != "mimetype":
            raise RuntimeError(
                "완성 HWPX mimetype 위치 오류"
            )

        if check.read("mimetype") != b"application/hwp+zip":
            raise RuntimeError(
                "완성 HWPX mimetype 오류"
            )

        if check.testzip() is not None:
            raise RuntimeError(
                "완성 HWPX ZIP 무결성 검사 실패"
            )

        for name in check.namelist():

            if name.lower().endswith(".xml"):
                ET.fromstring(
                    check.read(name).decode("utf-8")
                )

    return output_path

def create_hwpx(
    template_path,
    output_path,
    title,
    body,
    notice_date,
    notice_deadline,
    subject,
    work_date,
    company,
    phone,
    office
):
    return create_notice_hwpx(
        template_path,
        output_path,
        title,
        body,
        notice_date,
        notice_deadline,
        subject,
        work_date,
        company,
        phone,
        office
    )


def show_notice_generator():
    st.markdown(
        '<div class="ai-avatar" style="font-size:55px;">📄</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="ai-greeting" style="font-size:21px;">AI 안내문 생성</div>',
        unsafe_allow_html=True
    )

    st.info(
        "건명과 안내문 요청을 입력하면 등록된 규정집에서 관련 내용을 "
        "먼저 검색하여 규정 중심의 안내문을 최대 5줄로 작성합니다."
    )

    notice_date = st.text_input(
        "① 공고일자",
        placeholder="예: 2026년 9월 1일",
        key="notice_date"
    )

    notice_deadline = st.text_input(
        "② 공고기한",
        placeholder="예: 2026년 9월 10일까지",
        key="notice_deadline"
    )

    subject = st.text_input(
        "③ 건명",
        placeholder="예: 보일러 세관",
        key="notice_subject"
    )

    work_date = st.text_input(
        "④ 일시",
        placeholder="예: 2026년 9월 10일 09:00~17:00",
        key="notice_work_date"
    )

    company = st.text_input(
        "⑤ 업체",
        placeholder="예: ○○설비",
        key="notice_company"
    )

    phone = st.text_input(
        "⑥ 전화번호",
        placeholder="예: 053-123-4567",
        key="notice_phone"
    )

    office = st.text_input(
        "⑦ 관리소명",
        placeholder="예: ○○관리소",
        key="notice_office"
    )

    request_text = st.text_area(
        "⑧ 안내문 요청",
        placeholder=(
            "예: 보일러 세관에 대한 안내문을 "
            "관련 규정 위주로 5줄 이내 작성해줘"
        ),
        height=120,
        key="notice_request"
    )

    if st.button(
        "✨ 안내문 생성",
        key="notice_create_unique",
        use_container_width=True
    ):

        missing = []

        for label, value in [
            ("공고일자", notice_date),
            ("건명", subject),
            ("일시", work_date),
            ("업체", company),
            ("전화번호", phone),
            ("관리소명", office),
            ("안내문 요청", request_text),
        ]:
            if not str(value).strip():
                missing.append(label)

        if missing:
            st.warning(
                "다음 항목을 입력해주세요: "
                + ", ".join(missing)
            )
            return

        try:

            # =================================================
            # 1. templates/plan 규정집을 최우선 검색
            # =================================================
            plan_results = search_plan_regulations(
                query=f"{subject} {request_text}",
                top_k=5,
                min_score=0.25
            )

            # PLAN에서 찾은 근거가 우선.
            # PLAN에서 찾지 못한 경우 기존 vector_db도 보조적으로 사용.
            regulation_results = list(
                plan_results
            )

            if not regulation_results:

                for filename in filenames:

                    regulation_results.extend(
                        search_documents(
                            f"{subject} {request_text}",
                            filename,
                            top_k=2
                        )
                    )

                regulation_results.sort(
                    key=lambda x: float(
                        x.get("score", 0)
                    ),
                    reverse=True
                )

            context_parts = []

            for i, result in enumerate(
                regulation_results[:8],
                start=1
            ):

                source_type = (
                    "PLAN 규정집(plan.pdf)"
                    if result.get("filename") == "plan.pdf"
                    else
                    "기존 규정집"
                )

                context_parts.append(
                    f"[검색결과 {i}]\n"
                    f"출처: {source_type}\n"
                    f"문서명: {result.get('filename', '')}\n"
                    f"페이지: {result.get('page', '')}\n"
                    f"내용: {result.get('text', '')}"
                )

            regulation_context = "\n\n".join(
                context_parts
            )

            with st.spinner(
                "🤖 관련 규정을 확인하고 안내문을 작성하고 있습니다..."
            ):

                title, body, basis = (
                    generate_notice_text(
                        request_text,
                        subject,
                        work_date,
                        company,
                        phone,
                        office,
                        regulation_context
                    )
                )

            st.success("✅ 안내문 문구가 완성되었습니다.")

            st.markdown(
                '<div class="section-title">📝 생성된 안내문</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="welcome-card">
                <div class="welcome-title">제목</div>
                <div style="font-size:18px;font-weight:800;
                color:#243c56;margin-bottom:15px;">
                {html.escape(title)}
                </div>

                <div class="welcome-title">안내내용</div>
                <div style="font-size:14px;line-height:2;
                color:#334b64;white-space:pre-line;">
                {html.escape(body).replace(chr(10), "<br>")}
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if basis:
                st.caption(
                    "📚 참고한 근거: " + basis
                )

            with st.spinner(
                "📄 한글파일을 생성하고 있습니다..."
            ):

                # GitHub에서 원본 양식 다운로드
                template_file = download_hwpx_template(
                    HWPX_TEMPLATE_URLS
                )

                output_file = os.path.join(
                    tempfile.gettempdir(),
                    "안내문_완성본.hwpx"
                )

                create_hwpx(
                    template_file,
                    output_file,
                    title,
                    body,
                    notice_date,
                    notice_deadline,
                    subject,
                    work_date,
                    company,
                    phone,
                    office
                )

            with open(
                output_file,
                "rb"
            ) as f:
                data = f.read()

            st.download_button(
                "📥 완성된 안내문 한글파일 다운로드",
                data=data,
                file_name=f"{subject}_안내문.hwpx",
                mime="application/vnd.hancom.hwpx",
                key="notice_download_unique",
                use_container_width=True
            )

        except Exception as e:

            err = str(e)
            upper = err.upper()

            if (
                "503" in upper
                or "UNAVAILABLE" in upper
                or "429" in upper
                or "RESOURCE_EXHAUSTED" in upper
            ):
                st.warning(
                    "🤖 Gemini 서버가 현재 혼잡합니다. "
                    "잠시 후 다시 생성해주세요."
                )
            else:
                st.error(
                    "❌ 안내문 생성 중 오류가 발생했습니다."
                )
                st.code(
                    err,
                    language="text"
                )

    if st.button(
        "↩️ 처음 화면으로 돌아가기",
        key="notice_back_unique",
        use_container_width=True
    ):
        st.session_state.show_notice_generator = False
        st.rerun()


# =========================================================
# 13. 첫 화면
# =========================================================

if st.session_state.get("show_notice_generator", False):

    show_notice_generator()
    st.stop()


if st.session_state.selected_file is None:

    # -----------------------------------------------------
    # AI 비서 아이콘
    # -----------------------------------------------------

    st.markdown(
        '<div class="ai-avatar">🤖</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # 인사말
    # -----------------------------------------------------

    st.markdown(
        '<div class="ai-greeting">'
        '대구경북지사 직원 여러분 안녕하세요^^<br>'
        '무엇을 도와드릴까요?'
        '</div>',
        unsafe_allow_html=True
    )


  


    # -----------------------------------------------------
    # 안내 카드
    # -----------------------------------------------------

    st.markdown(
        """
        <div class="welcome-card">

        <div class="welcome-title">
        🤖 규정에 근거하여 답변합니다
        </div>

        <div class="welcome-text">
        등록된 사내 규정집에서 질문과 관련성이 높은
        내용을 찾아 답변해 드립니다.<br>
        <b>선택한 규정집에 명시되지 않은 내용은
        임의로 답변하지 않습니다.</b>
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


 


    # -----------------------------------------------------
    # 안내문 생성
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📝 업무 지원</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "📄 안내문 생성",
        key="notice_generator_home_unique",
        use_container_width=True
    ):
        st.session_state.show_notice_generator = True
        st.rerun()


    # -----------------------------------------------------
    # 규정집 선택
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📚 먼저 규정집을 선택해주세요</div>',
        unsafe_allow_html=True
    )


    for i, filename in enumerate(filenames):

        if st.button(
            f"📄  {filename}",
            key=f"pdf_{i}",
            use_container_width=True
        ):

            st.session_state.selected_file = filename

            st.session_state.messages = []

            st.rerun()


# =========================================================
# 14. 규정집 선택 후 화면
# =========================================================

else:

    selected_file = st.session_state.selected_file


    # -----------------------------------------------------
    # AI 비서 아이콘
    # -----------------------------------------------------

    st.markdown(
        '<div class="ai-avatar" style="font-size:55px;">🤖</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="ai-greeting" style="font-size:21px;">'
        '무엇이 궁금하신가요?'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # 현재 선택된 규정집
    # -----------------------------------------------------

    st.markdown(
        f"""
        <div class="selected-document">

        <div class="selected-document-title">
        📚 현재 선택된 규정집
        </div>

        <div class="selected-document-name">
        {selected_file}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # 규정집 목록으로 돌아가기
    # -----------------------------------------------------

    if st.button(
        "↩️ 다른 규정집 선택",
        key="back_to_documents",
        use_container_width=True
    ):

        st.session_state.selected_file = None

        st.session_state.messages = []

        st.rerun()


   

    # -----------------------------------------------------
    # 이전 대화 표시
    # -----------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # -----------------------------------------------------
    # 질문 입력
    # -----------------------------------------------------

    user_input = st.chat_input(
        "궁금한 내용을 입력하세요..."
    )


    # =====================================================
    # 질문이 들어온 경우
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
                # ① 검색
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
                        label="관련 규정을 찾지 못했습니다.",
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
                    # ③ 검색 결과 확인
                    # =================================

                    status.update(
                        label=(
                            f"관련 규정 "
                            f"{len(search_results)}건을 찾았습니다."
                        ),
                        state="running",
                        expanded=True
                    )


                    # =================================
                    # ④ 검색 내용 정리
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

사용자가 선택한 규정집의 검색 결과만을
근거로 답변해야 한다.

[선택된 규정집]

{st.session_state.selected_file}

[답변 원칙]

1. 검색된 규정 문서의 내용을 근거로 답변한다.

2. 검색된 내용에 없는 사항은 추측하지 않는다.

3. 일반적인 지식이나 인터넷 정보를 사용하지 않는다.

4. 근거가 부족한 경우 다음 문장으로 답변한다.

"해당 내용은 선택한 규정집에 명시되어 있지 않습니다."

5. 답변의 근거가 되는 문서명과 페이지를 표시한다.

6. 규정이나 업무절차는 이해하기 쉽게
번호나 항목으로 정리한다.

7. 검색 결과가 서로 다른 경우
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


                    # =================================
                    # ⑥ Gemini 호출
                    # =================================

                    status.update(
                        label=(
                            "🤖 규정을 확인했습니다. "
                            "답변을 작성하고 있습니다..."
                        ),
                        state="running",
                        expanded=True
                    )


                    answer = None


                    for attempt in range(3):

                        try:

                            response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
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
                # ⑧ 답변 저장
                # =================================

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
                    f"{str(e)}"
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
