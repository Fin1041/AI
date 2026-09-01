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

# =========================================================
# 13-A. AI 안내문 생성
# =========================================================

# GitHub 직접 설정
GITHUB_USERNAME = "Fin1041"
GITHUB_REPOSITORY = "AI"
GITHUB_BRANCH = "main"

PLAN_PDF_PATH = "templates/plan.pdf"
HWPX_TEMPLATE_PATH = "templates/notice_template.hwpx"


def _github_headers():
    return {
        "User-Agent": "house-management-notice-app"
    }


def _github_raw_url(path):
    return (
        "https://raw.githubusercontent.com/"
        f"{GITHUB_USERNAME}/"
        f"{GITHUB_REPOSITORY}/"
        f"{GITHUB_BRANCH}/"
        f"{path}"
    )


HWPX_TEMPLATE_URL = _github_raw_url(
    HWPX_TEMPLATE_PATH
)

PLAN_PDF_URL = _github_raw_url(
    PLAN_PDF_PATH
)


def download_bytes(url):
    request = urllib.request.Request(
        url,
        headers=_github_headers()
    )

    with urllib.request.urlopen(
        request,
        timeout=30
    ) as response:
        return response.read()


def download_hwpx_template(url):
    try:
        data = download_bytes(url)
    except Exception as e:
        raise RuntimeError(
            "GitHub에서 notice_template.hwpx를 가져오지 못했습니다.\n"
            + str(e)
        )

    try:
        with zipfile.ZipFile(
            __import__("io").BytesIO(data),
            "r"
        ) as z:
            names = z.namelist()

            if not names or names[0] != "mimetype":
                raise RuntimeError(
                    "다운로드한 파일이 정상적인 HWPX가 아닙니다."
                )

            if z.read("mimetype") != b"application/hwp+zip":
                raise RuntimeError(
                    "HWPX mimetype이 올바르지 않습니다."
                )

            if "Contents/section0.xml" not in names:
                raise RuntimeError(
                    "HWPX에 Contents/section0.xml이 없습니다."
                )

    except zipfile.BadZipFile as e:
        raise RuntimeError(
            "GitHub에서 HWPX가 아닌 파일을 받았습니다."
        ) from e

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


@st.cache_data(ttl=3600, show_spinner=False)
def load_plan_pdf_pages():
    """
    GitHub의 templates/plan.pdf 단일 규정집을
    페이지 단위로 읽는다.
    """
    try:
        data = download_bytes(
            PLAN_PDF_URL
        )
    except Exception as e:
        raise RuntimeError(
            "templates/plan.pdf를 가져오지 못했습니다.\n"
            + str(e)
        )

    try:
        reader = PdfReader(
            __import__("io").BytesIO(data)
        )
    except Exception as e:
        raise RuntimeError(
            "plan.pdf를 읽지 못했습니다.\n"
            + str(e)
        )

    pages = []

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
            text = ""

        if text:
            pages.append(
                {
                    "page": page_no,
                    "text": text
                }
            )

    return pages


@st.cache_resource
def load_plan_search_index():
    """
    plan.pdf 페이지를 기존 embedding 모델로 인덱싱한다.
    """
    pages = load_plan_pdf_pages()

    if not pages:
        return None, []

    embeddings = embedding_model.encode(
        [
            "passage: " + page["text"][:6000]
            for page in pages
        ],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    return embeddings, pages


def search_plan_pdf(
    query,
    top_k=5
):
    """
    안내문 건명/요청내용과 관련된 plan.pdf 근거를 검색한다.
    """
    query = str(
        query or ""
    ).strip()

    if not query:
        return []

    embeddings, pages = (
        load_plan_search_index()
    )

    if embeddings is None:
        return []

    query_embedding = embedding_model.encode(
        ["query: " + query],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )[0]

    scores = (
        embeddings @ query_embedding
    )

    order = np.argsort(
        scores
    )[::-1]

    results = []

    for idx in order[:top_k]:

        results.append(
            {
                "filename": "plan.pdf",
                "page": pages[int(idx)]["page"],
                "text": pages[int(idx)]["text"],
                "score": float(
                    scores[idx]
                )
            }
        )

    return results


def generate_notice_text(
    request_text,
    subject,
    plan_context=""
):
    """
    제목 15자 이하 / 본문 무조건 5줄 이하.
    plan.pdf에 실제 근거가 있으면 본문 안에 직접 포함한다.
    """

    prompt = f"""
너는 공동주택 관리사무소의 공식 안내문 작성 담당자이다.

[건명]
{subject}

[사용자 요청]
{request_text}

[plan.pdf에서 검색한 관련 규정]
{plan_context if plan_context else "관련 규정 근거를 찾지 못했습니다."}

작성 규칙:
1. 제목은 반드시 15자 이하로 작성한다.
2. 제목은 건명의 핵심어를 포함한다.
3. 안내내용은 반드시 5줄 이내다.
4. 한 줄에 한 문장만 작성한다.
5. 각 줄은 가능하면 공백 제외 약 20~25글자 정도로 간결하게 쓴다.
6. 내용은 건명에 대한 목적, 필요성, 관리방법, 입주민 협조 순으로 간단히 작성한다.
7. plan.pdf에서 실제 관련 규정이 확인되면 반드시 안내내용 중 1개 문장에
   규정명 또는 확인된 기준을 자연스럽게 직접 포함한다.
8. 규정은 별도의 참고사항처럼 길게 쓰지 말고 본문 문장 속에 넣는다.
9. 검색결과에 없는 법조문 번호나 법적 의무를 절대로 만들어내지 않는다.
10. plan.pdf에서 관련 근거가 없으면 규정을 억지로 넣지 않는다.
11. 일시, 날짜, 업체명, 전화번호, 관리소명은 본문에 넣지 않는다.
12. 같은 표현을 반복하지 않는다.
13. 입주민이 한눈에 이해할 수 있는 쉬운 표현을 사용한다.
14. 본문은 최대 5줄을 절대 넘기지 않는다.

[출력형식]
[제목]
제목

[본문]
문장1
문장2
문장3
문장4
문장5

[근거]
plan.pdf에서 실제 확인된 규정명과 페이지만 간단히 표시
"""

    last_error = None

    for attempt in range(5):

        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt
            )

            text = (
                response.text
                or ""
            ).strip()

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
                time.sleep(
                    2 ** attempt
                )

    if not text:
        raise RuntimeError(
            "Gemini 서버가 현재 응답하지 않습니다.\n"
            + str(last_error)
        )

    title = ""
    body = ""
    basis = ""

    if "[제목]" in text:

        rest = text.split(
            "[제목]",
            1
        )[1]

        if "[본문]" in rest:

            title, rest = rest.split(
                "[본문]",
                1
            )

            if "[근거]" in rest:

                body, basis = rest.split(
                    "[근거]",
                    1
                )

            else:
                body = rest

        else:
            title = rest

    title = re.sub(
        r"\s+",
        " ",
        title.strip()
    )[:15]

    if not title:
        title = (
            subject.strip()
            or "안내문"
        )[:15]

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
    ][:5]

    body = "\n".join(lines)

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
    {{안내내용1}}~{{안내내용5}}에
    각각 한 문장씩 넣는다.
    """

    body_lines = [
        line.strip()
        for line in str(body).splitlines()
        if line.strip()
    ][:5]

    # 짧은 본문을 위해 최대 5개 paragraph 사용
    with zipfile.ZipFile(
        template_path,
        "r"
    ) as zin:

        names = zin.namelist()

        if not names or names[0] != "mimetype":
            raise RuntimeError(
                "HWPX 원본 구조가 올바르지 않습니다."
            )

        section = "Contents/section0.xml"

        if section not in names:
            raise RuntimeError(
                "Contents/section0.xml이 없습니다."
            )

        data = {
            name: zin.read(name)
            for name in names
        }

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

        # 전체 XML 일반 치환
        for name in names:

            if not name.lower().endswith(".xml"):
                continue

            xml_text = data[name].decode(
                "utf-8"
            )

            for key, value in replacements.items():

                xml_text = xml_text.replace(
                    key,
                    html.escape(
                        str(value),
                        quote=False
                    )
                )

            data[name] = xml_text.encode(
                "utf-8"
            )

        xml = data[section].decode(
            "utf-8"
        )

        # 번호형 안내내용1~5
        for i in range(1, 6):

            key = f"{{{{안내내용{i}}}}}"

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
                    f"{key} 위치를 찾지 못했습니다."
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
                    f"{key}의 기존 텍스트 영역을 찾지 못했습니다."
                )

            value = (
                body_lines[i - 1]
                if i <= len(body_lines)
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

        # XML 검증
        ET.fromstring(xml)

        data[section] = xml.encode(
            "utf-8"
        )

        # ZIP 구조 유지
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

    # 최종 검사
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
                    check.read(name).decode(
                        "utf-8"
                    )
                )

    return output_path


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
        "plan.pdf의 관련 규정을 먼저 확인하여 "
        "근거가 있으면 안내내용에 직접 포함합니다."
    )

    notice_date = st.text_input(
        "① 공고일자",
        key="notice_date_direct_reg"
    )

    notice_deadline = st.text_input(
        "② 공고기한",
        key="notice_deadline_direct_reg"
    )

    subject = st.text_input(
        "③ 건명",
        placeholder="예: 보일러 세관",
        key="notice_subject_direct_reg"
    )

    work_date = st.text_input(
        "④ 일시",
        key="notice_work_date_direct_reg"
    )

    company = st.text_input(
        "⑤ 업체",
        key="notice_company_direct_reg"
    )

    phone = st.text_input(
        "⑥ 전화번호",
        key="notice_phone_direct_reg"
    )

    office = st.text_input(
        "⑦ 관리소명",
        key="notice_office_direct_reg"
    )

    request_text = st.text_area(
        "⑧ 안내문 요청",
        placeholder=(
            "예: 보일러 세관에 대한 안내문을 "
            "관련 규정 중심으로 작성해줘"
        ),
        height=110,
        key="notice_request_direct_reg"
    )

    if st.button(
        "📄 안내문 생성",
        key="notice_create_direct_reg",
        use_container_width=True
    ):

        required = [
            ("공고일자", notice_date),
            ("건명", subject),
            ("일시", work_date),
            ("업체", company),
            ("전화번호", phone),
            ("관리소명", office),
            ("안내문 요청", request_text),
        ]

        missing = [
            name
            for name, value in required
            if not str(value).strip()
        ]

        if missing:

            st.warning(
                "다음 항목을 입력해주세요: "
                + ", ".join(missing)
            )

            return

        try:

            # ① plan.pdf 우선 검색
            with st.spinner(
                "📚 plan.pdf에서 관련 규정을 찾고 있습니다..."
            ):

                plan_results = search_plan_pdf(
                    f"{subject} {request_text}",
                    top_k=5
                )

            context_parts = []

            for i, result in enumerate(
                plan_results,
                start=1
            ):

                context_parts.append(
                    f"[규정근거 {i}]\n"
                    f"문서명: plan.pdf\n"
                    f"페이지: {result['page']}페이지\n"
                    f"내용:\n{result['text']}"
                )

            plan_context = "\n\n".join(
                context_parts
            )

            # ② Gemini 안내문 작성
            with st.spinner(
                "🤖 규정 근거를 반영하여 안내문을 작성하고 있습니다..."
            ):

                title, body, basis = (
                    generate_notice_text(
                        request_text=request_text,
                        subject=subject,
                        plan_context=plan_context
                    )
                )

            st.success(
                "✅ 안내문이 완성되었습니다."
            )

            st.markdown(
                f"""
                <div class="welcome-card">
                <div class="welcome-title">제목</div>
                <div style="font-size:18px;font-weight:800;
                margin-bottom:15px;">
                {html.escape(title)}
                </div>

                <div class="welcome-title">안내내용</div>
                <div style="font-size:14px;line-height:2;
                white-space:pre-line;">
                {html.escape(body)}
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if basis:
                st.caption(
                    "📚 참고한 근거: " + basis
                )

            # ③ HWPX 생성
            with st.spinner(
                "📄 한글파일을 생성하고 있습니다..."
            ):

                template_file = download_hwpx_template(
                    HWPX_TEMPLATE_URL
                )

                output_file = os.path.join(
                    tempfile.gettempdir(),
                    "안내문_완성본.hwpx"
                )

                create_notice_hwpx(
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
                hwpx_data = f.read()

            st.download_button(
                "📥 완성된 한글파일 다운로드",
                data=hwpx_data,
                file_name=f"{subject}_안내문.hwpx",
                mime="application/vnd.hancom.hwpx",
                key="notice_download_direct_reg",
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
        key="notice_back_direct_reg",
        use_container_width=True
    ):

        st.session_state.show_notice_generator = False
        st.rerun()


if "show_notice_generator" not in st.session_state:
    st.session_state.show_notice_generator = False

# =========================================================
# 13. 첫 화면
# =========================================================

if st.session_state.get("show_notice_generator", False):

    show_notice_generator()
    st.stop()


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
        key="notice_legacy_removed",
        use_container_width=True
    ):
        st.session_state.show_notice_generator = True
        st.rerun()


    # -----------------------------------------------------
    # 업무 지원
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📝 업무 지원</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "📄 안내문 생성",
        key="notice_home_direct_reg",
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
