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
from pypdf import PdfReader


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

if "rulebook_category" not in st.session_state:
    st.session_state.rulebook_category = None


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

GITHUB_USERNAME = "Fin1041"
GITHUB_REPOSITORY = "AI"
GITHUB_BRANCH = "main"
HWPX_TEMPLATE_PATH = "templates/notice_template.hwpx"

# 안내문에는 plan.pdf를 법적 근거로 사용하지 않는다.
# AI가 공식 법령정보를 검색하도록 별도 검색 단계를 둔다.
LAW_SEARCH_URL = "https://www.law.go.kr/"

HWPX_TEMPLATE_URL = (
    f"https://raw.githubusercontent.com/"
    f"{GITHUB_USERNAME}/{GITHUB_REPOSITORY}/"
    f"{GITHUB_BRANCH}/{HWPX_TEMPLATE_PATH}"
)


def _github_download(url):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "house-management-notice-app"}
    )

    with urllib.request.urlopen(
        request,
        timeout=30
    ) as response:
        return response.read()


@st.cache_data(ttl=3600, show_spinner=False)
def get_notice_template_bytes():
    """
    GitHub의 HWPX 양식을 1시간 캐시한다.
    안내문을 만들 때마다 다시 다운로드하지 않도록 해 속도를 높인다.
    """
    data = _github_download(HWPX_TEMPLATE_URL)

    try:
        with zipfile.ZipFile(__import__("io").BytesIO(data), "r") as z:
            names = z.namelist()
            if not names or names[0] != "mimetype":
                raise RuntimeError(
                    "notice_template.hwpx가 정상적인 HWPX가 아닙니다."
                )
            if z.read("mimetype") != b"application/hwp+zip":
                raise RuntimeError(
                    "HWPX mimetype이 올바르지 않습니다."
                )
            if "Contents/section0.xml" not in names:
                raise RuntimeError(
                    "HWPX의 section0.xml을 찾지 못했습니다."
                )
            section = z.read("Contents/section0.xml").decode("utf-8")
            for i in range(1, 6):
                if f"{{{{안내내용{i}}}}}" not in section:
                    raise RuntimeError(
                        f"notice_template.hwpx에 {{{{안내내용{i}}}}}이 없습니다."
                    )
    except zipfile.BadZipFile as e:
        raise RuntimeError(
            "GitHub에서 받은 notice_template.hwpx가 손상되었습니다."
        ) from e

    return data


def download_notice_template():
    data = get_notice_template_bytes()

def search_official_law_with_ai(
    subject,
    request_text
):
    """
    AI에게 최신 공식 법령 근거를 찾도록 요청한다.
    검색 결과가 확인되지 않으면 법조문을 만들어내지 않는다.

    주의: Gemini 자체가 웹 브라우징 도구를 제공하지 않는 환경에서는
    모델 지식만으로 확정하지 않고 '근거 미확인'으로 처리한다.
    """

    law_prompt = f"""
너는 공동주택 관리사무소의 법령 근거 확인 담당자다.

건명: {subject}
요청: {request_text}

현행 대한민국 법령을 대상으로 안내문에 넣을 수 있는 근거를 판단한다.
law.go.kr 기준을 우선하되, 실제 확인하지 못한 법령명·조문번호·시행일은 쓰지 않는다.
근거가 확실하지 않으면 반드시 '근거 미확인'으로 표시한다.
확인된 경우 법령명, 조문번호, 핵심근거만 짧게 작성한다.

출력:
[법령명]
[조문]
[핵심근거]
[확인상태] 확인 / 근거 미확인
"""

    last_error = None

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=law_prompt
            )

            text = (
                response.text
                or ""
            ).strip()

            if text:
                return text

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

            if attempt < 2:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        "법령 근거 확인을 위한 AI 호출에 실패했습니다.\n"
        + str(last_error)
    )


def generate_notice_text_with_law(
    request_text,
    subject,
    law_context
):
    """
    제목 15자 이하 + 최대 5줄.
    법령 근거가 실제 확인된 경우 본문 1줄에 직접 포함.
    """

    prompt = f"""
너는 공동주택 관리사무소의 공식 안내문 작성 담당자다.

건명: {subject}
요청: {request_text}
법령 근거: {law_context if law_context else "확인된 법령 근거 없음"}

규칙: 제목 15자 이하, 본문 최대 5줄. 각 줄은 짧고 읽기 쉽게 작성한다. 건명→필요성→법령근거→협조사항 순으로 구성한다. 확인된 법령만 사용하고 조문을 임의로 만들지 않는다. 일시·업체·전화번호·관리소명은 본문에서 반복하지 않는다.

출력:
[제목]
15자 이하
[본문]
최대 5줄
"""

    for attempt in range(3):
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

        except Exception as e:
            upper = str(e).upper()

            if not (
                "503" in upper
                or "UNAVAILABLE" in upper
                or "429" in upper
                or "RESOURCE_EXHAUSTED" in upper
            ):
                raise

            if attempt < 2:
                time.sleep(2 ** attempt)

    if not text:
        raise RuntimeError(
            "안내문 생성 결과가 없습니다."
        )

    title = ""
    body = ""

    if "[제목]" in text:
        rest = text.split("[제목]", 1)[1]

        if "[본문]" in rest:
            title, body = rest.split(
                "[본문]",
                1
            )
        else:
            title = rest

    title = re.sub(
        r"\s+",
        " ",
        title.strip()
    )[:15]

    body = body.strip()

    # 번호/불릿 제거
    body = re.sub(
        r"^\s*(?:[-•·]|\d+[\.\)])\s*",
        "",
        body,
        flags=re.MULTILINE
    )

    lines = [
        x.strip()
        for x in body.splitlines()
        if x.strip()
    ][:5]

    return title or (subject[:15] or "안내문"), lines


def create_notice_hwpx(
    template_path,
    output_path,
    title,
    lines,
    notice_date,
    notice_deadline,
    custom_fields,
    phone,
    office
):
    """
    현재 notice_template.hwpx의 구조에 맞춰 안전하게 생성한다.

    실제 양식 구조:
      {{제목}}
      {{안내내용1}} ~ {{안내내용5}}
      {{항목명}} {{입력내용}}  x 5
      {{전화번호}}
      {{관리소명}}

    같은 placeholder가 5번 반복되므로, 문단 단위로 순서대로
    1~5번 항목을 치환한다.
    """

    lines = list(lines or [])[:5]
    custom_fields = list(custom_fields or [])[:5]

    if len(custom_fields) > 5:
        raise RuntimeError("사용자 지정 항목은 최대 5개까지 입력할 수 있습니다.")

    with zipfile.ZipFile(template_path, "r") as zin:
        names = zin.namelist()

        if not names or names[0] != "mimetype":
            raise RuntimeError("원본 HWPX 구조가 올바르지 않습니다.")

        if zin.read("mimetype") != b"application/hwp+zip":
            raise RuntimeError("원본 HWPX mimetype이 올바르지 않습니다.")

        data = {name: zin.read(name) for name in names}

        replacements = {
            "{{공고일자}}": notice_date,
            "{{공고기한}}": notice_deadline,
            "{{제목}}": str(title)[:15],
            "{{전화번호}}": phone,
            "{{관리소명}}": office,
        }

        # 일반 placeholder 치환
        for name in names:
            if not name.lower().endswith(".xml"):
                continue

            try:
                xml = data[name].decode("utf-8")
            except UnicodeDecodeError:
                continue

            for key, value in replacements.items():
                xml = xml.replace(
                    key,
                    html.escape(str(value), quote=False)
                )

            data[name] = xml.encode("utf-8")

        section = "Contents/section0.xml"
        if section not in data:
            raise RuntimeError("HWPX에서 Contents/section0.xml을 찾지 못했습니다.")

        xml = data[section].decode("utf-8")

        # -------------------------------------------------
        # 1~5번 사용자 항목
        # 현재 양식에는 동일한 {{항목명}} / {{입력내용}}이
        # 5개 문단에 반복되어 있으므로 각 문단을 순서대로 치환한다.
        # -------------------------------------------------
        paragraphs = list(
            re.finditer(
                r"<hp:p\b[^>]*>(?:(?!<hp:p\b)[\s\S])*?</hp:p>",
                xml,
                re.DOTALL
            )
        )

        target_indices = []
        for m in paragraphs:
            paragraph = m.group(0)
            if "{{항목명}}" in paragraph and "{{입력내용}}" in paragraph:
                target_indices.append(m)

        if len(target_indices) < 5:
            raise RuntimeError(
                "HWPX에서 {{항목명}}/{{입력내용}} 5개 항목 위치를 모두 찾지 못했습니다."
            )

        for idx in range(5):
            match = target_indices[0]
            paragraph = match.group(0)
            if idx < len(custom_fields):
                label, value = custom_fields[idx]
                safe_label = html.escape(str(label), quote=False)
                safe_value = html.escape(str(value), quote=False)
            else:
                # 입력하지 않은 나머지 행은 문단 전체를 비워 표시되지 않게 한다.
                label = ""
                value = ""
                safe_label = ""
                safe_value = ""

            # 같은 문단의 두 placeholder를
            # "1. 항목명 : 입력내용" 형식으로 표시한다.
            if label and value:
                display_text = f"{idx + 1}. {label} : {value}"
                safe_display = html.escape(display_text, quote=False)
                new_paragraph = paragraph.replace("{{항목명}} {{입력내용}}", safe_display, 1)
                new_paragraph = new_paragraph.replace("{{항목명}}", safe_label, 1)
                new_paragraph = new_paragraph.replace("{{입력내용}}", safe_value, 1)
            else:
                # 입력하지 않은 행은 실제 표시가 남지 않도록 완전히 비운다.
                new_paragraph = paragraph.replace("{{항목명}}", "", 1)
                new_paragraph = new_paragraph.replace("{{입력내용}}", "", 1)

            # 원본 xml에서 뒤쪽부터 바꿔 위치가 틀어지지 않도록
            # 치환 대상 5개를 뒤에서부터 적용한다.
            xml = (
                xml[:match.start()]
                + new_paragraph
                + xml[match.end():]
            )

            # 앞의 치환으로 match 위치가 변했으므로 다음 반복에서
            # 다시 현재 XML 기준으로 항목 문단을 찾는다.
            if idx < 4:
                paragraphs = list(
                    re.finditer(
                        r"<hp:p\b[^>]*>(?:(?!<hp:p\b)[\s\S])*?</hp:p>",
                        xml,
                        re.DOTALL
                    )
                )
                target_indices = [
                    m for m in paragraphs
                    if "{{항목명}}" in m.group(0) and "{{입력내용}}" in m.group(0)
                ]

        # -------------------------------------------------
        # 안내내용1~5
        # 기존 문단과 run 구조를 유지하면서 텍스트만 교체한다.
        # -------------------------------------------------
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
                raise RuntimeError(f"{key} 문단을 찾지 못했습니다.")

            paragraph = p.group(0)

            t = re.search(
                r"<hp:t\b[^>]*>.*?"
                + re.escape(key)
                + r".*?</hp:t>",
                paragraph,
                re.DOTALL
            )

            if not t:
                raise RuntimeError(f"{key}의 기존 텍스트 영역을 찾지 못했습니다.")

            value = lines[i - 1] if i <= len(lines) else ""
            new_t = (
                "<hp:t>"
                + html.escape(str(value), quote=False)
                + "</hp:t>"
            )

            paragraph = paragraph[:t.start()] + new_t + paragraph[t.end():]
            xml = xml[:p.start()] + paragraph + xml[p.end():]

        # 수정된 XML 자체 검증
        try:
            ET.fromstring(xml)
        except ET.ParseError as e:
            raise RuntimeError("수정된 HWPX XML 검증 실패: " + str(e))

        # 모든 placeholder가 남아있는지 확인
        required_markers = [
            "{{제목}}",
            "{{안내내용1}}", "{{안내내용2}}", "{{안내내용3}}",
            "{{안내내용4}}", "{{안내내용5}}",
            "{{항목명}}", "{{입력내용}}",
            "{{전화번호}}", "{{관리소명}}"
        ]

        # {{항목명}}, {{입력내용}}은 0개여야 한다.
        if "{{항목명}}" in xml or "{{입력내용}}" in xml:
            raise RuntimeError("사용자 지정 항목 치환이 완료되지 않았습니다.")

        for marker in [
            "{{제목}}",
            "{{안내내용1}}", "{{안내내용2}}", "{{안내내용3}}",
            "{{안내내용4}}", "{{안내내용5}}"
        ]:
            if marker in xml:
                raise RuntimeError(f"{marker} 치환이 완료되지 않았습니다.")

        data[section] = xml.encode("utf-8")

        # HWPX ZIP 구조 유지
        with zipfile.ZipFile(output_path, "w") as zout:
            for name in names:
                info = copy.copy(zin.getinfo(name))
                if name == "mimetype":
                    info.compress_type = zipfile.ZIP_STORED
                zout.writestr(info, data[name])

    # 최종 HWPX 무결성 검증
    with zipfile.ZipFile(output_path, "r") as check:
        if not check.namelist() or check.namelist()[0] != "mimetype":
            raise RuntimeError("완성 HWPX mimetype 위치 오류")
        if check.read("mimetype") != b"application/hwp+zip":
            raise RuntimeError("완성 HWPX mimetype 오류")
        if check.testzip() is not None:
            raise RuntimeError("완성 HWPX ZIP 무결성 검사 실패")

        final_xml = check.read(section).decode("utf-8")
        ET.fromstring(final_xml)

        # 실제 사용자 입력이 5개 모두 들어갔는지 확인
        for label, value in custom_fields:
            if str(label) not in final_xml or str(value) not in final_xml:
                raise RuntimeError(
                    f"완성 HWPX에 사용자 항목이 정상 반영되지 않았습니다: {label} / {value}"
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
        "관련 법령의 최신 근거를 확인하여 "
        "짧고 읽기 쉬운 안내문을 작성합니다. "
        "1~5번 항목명과 입력내용을 직접 설정할 수 있습니다."
    )

    notice_date = st.text_input(
        "① 공고일자",
        placeholder="예: 2026년 9월 1일",
        key="notice_date_law"
    )

    notice_deadline = st.text_input(
        "② 공고기한",
        placeholder="예: 2026년 9월 10일까지",
        key="notice_deadline_law"
    )

    st.markdown("**③~⑦ 안내문 항목을 사용자가 직접 지정할 수 있습니다.**")

    field1_label = ""
    field1_value = ""
    field2_label = ""
    field2_value = ""
    field3_label = ""
    field3_value = ""
    field4_label = ""
    field4_value = ""
    field5_label = ""
    field5_value = ""

    for idx, (label_key, value_key, label_example, value_example) in enumerate([
        ("notice_custom1_label_v3", "notice_custom1_value_v3", "예: 작업내용", "예: 보일러 세관"),
        ("notice_custom2_label_v3", "notice_custom2_value_v3", "예: 작업장소", "예: 지하 기계실"),
        ("notice_custom3_label_v3", "notice_custom3_value_v3", "예: 작업일시", "예: 2026년 9월 10일 09:00~17:00"),
        ("notice_custom4_label_v3", "notice_custom4_value_v3", "예: 담당부서", "예: 시설관리팀"),
        ("notice_custom5_label_v3", "notice_custom5_value_v3", "예: 문의사항", "예: 관리사무소로 문의"),
    ], start=1):
        col1, col2 = st.columns([1, 2])
        with col1:
            label_value = st.text_input(
                f"{idx}. 항목명",
                placeholder=label_example,
                key=label_key
            )
        with col2:
            value_value = st.text_input(
                f"{idx}. 입력내용",
                placeholder=value_example,
                key=value_key
            )

        if idx == 1:
            field1_label, field1_value = label_value, value_value
        elif idx == 2:
            field2_label, field2_value = label_value, value_value
        elif idx == 3:
            field3_label, field3_value = label_value, value_value
        elif idx == 4:
            field4_label, field4_value = label_value, value_value
        else:
            field5_label, field5_value = label_value, value_value

    phone = st.text_input(
        "⑥ 전화번호",
        placeholder="예: 053-123-4567",
        key="notice_phone_law"
    )

    office = st.text_input(
        "⑦ 관리소명",
        placeholder="예: ○○관리소",
        key="notice_office_law"
    )

    request_text = st.text_area(
        "⑧ 안내문 요청",
        placeholder=(
            "예: 보일러 세관 안내문을 작성해줘. "
            "관련 법규가 있으면 안내내용에 넣어줘."
        ),
        height=110,
        key="notice_request_law"
    )

    if st.button(
        "📄 안내문 생성",
        key="notice_create_law",
        use_container_width=True
    ):

        all_custom_fields = [
            (str(field1_label).strip(), str(field1_value).strip()),
            (str(field2_label).strip(), str(field2_value).strip()),
            (str(field3_label).strip(), str(field3_value).strip()),
            (str(field4_label).strip(), str(field4_value).strip()),
            (str(field5_label).strip(), str(field5_value).strip()),
        ]

        # 실제 입력이 완료된 항목만 사용한다.
        # 항목명과 입력내용 중 하나만 입력된 행은 오류로 안내한다.
        custom_fields = []
        partial_fields = []
        for idx, (label, value) in enumerate(all_custom_fields, 1):
            if label and value:
                custom_fields.append((label, value))
            elif label or value:
                partial_fields.append(f"{idx}번")

        missing = []
        if not notice_date.strip():
            missing.append("공고일자")
        if not notice_deadline.strip():
            missing.append("공고기한")

        if partial_fields:
            missing.append(" / ".join(partial_fields) + " 항목명과 입력내용 모두 입력")

        if not phone.strip():
            missing.append("전화번호")
        if not office.strip():
            missing.append("관리소명")
        if not request_text.strip():
            missing.append("안내문 요청")

        if missing:
            st.warning(
                "다음 항목을 입력해주세요: "
                + ", ".join(missing)
            )
            return

        try:

            with st.spinner(
                "⚖️ 관련 법령의 근거를 확인하고 있습니다..."
            ):
                field_summary = "\n".join(
                    f"{label}: {value}"
                    for label, value in custom_fields
                )
                subject = (
                    custom_fields[0][1]
                    if custom_fields
                    else "안내문"
                )
                law_context = search_official_law_with_ai(
                    subject,
                    request_text + "\n\n[안내문 기본항목]\n" + field_summary
                )

            with st.spinner(
                "🤖 확인된 법령 근거를 반영해 안내문을 작성하고 있습니다..."
            ):
                title, body_lines = (
                    generate_notice_text_with_law(
                        request_text + "\n\n[안내문 기본항목]\n" + field_summary,
                        subject,
                        law_context
                    )
                )

            # 화면에서는 실제 줄바꿈이 보이도록 Markdown 사용.
            st.success(
                "✅ 안내문이 완성되었습니다."
            )

            st.markdown(
                '<div class="section-title">📝 생성된 안내문</div>',
                unsafe_allow_html=True
            )

            safe_title = html.escape(title)
            safe_body = "<br>".join(
                html.escape(line)
                for line in body_lines
            )

            safe_fields = "<br>".join(
                html.escape(f"{idx}. {label} : {value}")
                for idx, (label, value) in enumerate(custom_fields, start=1)
            )

            fields_block = ""
            if safe_fields:
                fields_block = f"""
                <div class="welcome-title" style="margin-top:8px;">항목</div>
                <div style="font-size:14px;
                line-height:2.0;
                color:#334b64;
                margin-top:8px;
                margin-bottom:18px;">
                {safe_fields}
                </div>
                """

            st.markdown(
                f"""
                <div class="welcome-card">
                <div class="welcome-title">제목</div>
                <div style="font-size:18px;font-weight:800;
                color:#243c56;margin-bottom:18px;">
                {safe_title}
                </div>

                {fields_block}

                <div class="welcome-title">안내내용</div>
                <div style="font-size:14px;
                line-height:2.05;
                color:#334b64;
                margin-top:8px;">
                {safe_body}
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if (
                "근거 미확인" not in law_context
                and "확인상태" in law_context
            ):
                st.caption(
                    "⚖️ 법령 근거 확인 결과도 아래에서 확인할 수 있습니다."
                )
                with st.expander(
                    "확인된 법령 근거 보기"
                ):
                    st.write(law_context)

            with st.spinner(
                "📄 한글파일을 생성하고 있습니다..."
            ):

                template_file = (
                    download_notice_template()
                )

                output_file = os.path.join(
                    tempfile.gettempdir(),
                    "안내문_완성본.hwpx"
                )

                create_notice_hwpx(
                    template_file,
                    output_file,
                    title,
                    body_lines,
                    notice_date,
                    notice_deadline,
                    custom_fields,
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
                key="notice_download_law",
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
                    "🤖 AI 서버가 현재 혼잡합니다. 잠시 후 다시 시도해주세요."
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
        key="notice_back_law",
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


if st.session_state.selected_file is None:

    # -----------------------------------------------------
    # 첫 화면
    # -----------------------------------------------------

    if st.session_state.rulebook_category is None:

        # 첫 화면 전용 디자인
        st.markdown(
            """
            <style>
            /* 첫 화면에서만 기존 상단 제목/구분선 숨김 */
            .stApp:has(.home-screen) .top-title {
                display: none !important;
            }

            .stApp:has(.home-screen) hr {
                display: none !important;
            }

            .home-screen {
                width: 100%;
                margin: 0 auto;
            }

            /* 상단 네이비 헤더 */
            .home-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                background: #17304d;
                color: #ffffff;
                padding: 18px 28px;
                margin: -18px -18px 0 -18px;
                box-sizing: border-box;
            }

            .home-header-left {
                display: flex;
                align-items: center;
                gap: 12px;
                font-size: 24px;
                font-weight: 800;
                letter-spacing: -1.2px;
            }

            .home-header-house {
                font-size: 37px;
                line-height: 1;
            }

            .home-header-right {
                font-size: 16px;
                font-weight: 700;
                white-space: nowrap;
            }

            /* 중앙 제목 */
            .home-hero {
                text-align: center;
                padding: 34px 10px 29px;
            }

            .home-hero-title {
                margin: 0;
                color: #17365d;
                font-size: 44px;
                line-height: 1.15;
                font-weight: 900;
                letter-spacing: -2.3px;
            }

            .home-hero-title .ai-blue {
                color: #2777dd;
            }

            .home-hero-subtitle {
                margin-top: 13px;
                color: #334a67;
                font-size: 18px;
                line-height: 1.5;
                font-weight: 500;
                letter-spacing: -0.7px;
            }

            /* 바깥 테두리 */
            .home-panel-wrap {
                border: 2px solid #d7e0e9;
                border-radius: 24px;
                padding: 18px;
                background: #ffffff;
            }

            /* st.column 자체를 좌우 패널처럼 보이게 */
            div[data-testid="stColumn"]:has(.home-blue-marker) {
                background: linear-gradient(180deg, #edf6fd 0%, #e7f2fb 100%) !important;
                border-radius: 21px !important;
                padding: 24px 24px 21px !important;
                box-sizing: border-box;
            }

            div[data-testid="stColumn"]:has(.home-green-marker) {
                background: linear-gradient(180deg, #eef9ea 0%, #eaf6e7 100%) !important;
                border-radius: 21px !important;
                padding: 24px 24px 21px !important;
                box-sizing: border-box;
            }

            .home-blue-marker,
            .home-green-marker {
                display: none;
            }

            /* 패널 제목 */
            .home-panel-head {
                display: flex;
                align-items: center;
                gap: 16px;
                margin-bottom: 22px;
            }

            .home-panel-icon {
                width: 76px;
                height: 76px;
                min-width: 76px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 38px;
            }

            .home-blue-icon {
                background: #2b7cdd;
            }

            .home-green-icon {
                background: #24935c;
            }

            .home-panel-title {
                font-size: 27px;
                line-height: 1.2;
                font-weight: 900;
                letter-spacing: -1.3px;
            }

            .home-blue-title {
                color: #173f79;
            }

            .home-green-title {
                color: #176a42;
            }

            .home-panel-desc {
                margin-top: 6px;
                color: #334b67;
                font-size: 14px;
                line-height: 1.5;
                letter-spacing: -0.4px;
            }

            /* 메뉴 버튼 */
            .home-screen div.stButton {
                margin-top: 0 !important;
                margin-bottom: 12px !important;
            }

            .home-screen div.stButton > button {
                width: 100% !important;
                min-height: 112px !important;
                border-radius: 17px !important;
                border: 1px solid #d7e2ec !important;
                background: rgba(255,255,255,0.97) !important;
                box-shadow: 0 5px 13px rgba(39,71,103,0.08) !important;
                color: #18314f !important;
                font-size: 20px !important;
                font-weight: 900 !important;
                text-align: left !important;
                padding: 19px 22px !important;
                transition: 0.2s ease !important;
            }

            .home-screen div.stButton > button:hover {
                transform: translateY(-1px) !important;
                border-color: #b8cde1 !important;
                box-shadow: 0 7px 15px rgba(39,71,103,0.11) !important;
            }

            /* 버튼 아래 설명 */
            .home-menu-description {
                color: #405873;
                font-size: 13px;
                line-height: 1.5;
                margin: -1px 4px 13px;
                padding-left: 65px;
                position: relative;
                top: -54px;
                margin-bottom: -34px;
                pointer-events: none;
            }

            .home-menu-description::before {
                content: "";
                position: absolute;
                left: 16px;
                top: -5px;
                width: 46px;
                height: 46px;
                border-radius: 14px;
                background: #dcecfb;
            }

            .home-green-description::before {
                background: #e1f3e4;
            }

            /* 아이콘을 버튼 안에 함께 보이게 하기 위해 버튼 텍스트 앞쪽 공간 확보 */
            .home-blue-button-marker,
            .home-green-button-marker {
                display: none;
            }

            .home-menu-arrow {
                float: right;
                font-size: 32px;
                line-height: 1;
                margin-top: -2px;
                color: #236fc1;
            }

            /* 모바일 */
            @media (max-width: 780px) {

                .home-header {
                    padding: 15px 16px;
                    margin-left: -14px;
                    margin-right: -14px;
                }

                .home-header-left {
                    font-size: 18px;
                }

                .home-header-house {
                    font-size: 28px;
                }

                .home-header-right {
                    font-size: 13px;
                }

                .home-hero {
                    padding: 24px 4px 22px;
                }

                .home-hero-title {
                    font-size: 31px;
                    letter-spacing: -1.7px;
                }

                .home-hero-subtitle {
                    font-size: 13px;
                }

                .home-panel-wrap {
                    padding: 9px;
                    border-radius: 19px;
                }

                div[data-testid="stColumn"]:has(.home-blue-marker),
                div[data-testid="stColumn"]:has(.home-green-marker) {
                    padding: 18px 14px 17px !important;
                    border-radius: 18px !important;
                }

                .home-panel-head {
                    gap: 12px;
                    margin-bottom: 18px;
                }

                .home-panel-icon {
                    width: 58px;
                    height: 58px;
                    min-width: 58px;
                    font-size: 29px;
                }

                .home-panel-title {
                    font-size: 22px;
                }

                .home-panel-desc {
                    font-size: 12px;
                }

                .home-screen div.stButton > button {
                    min-height: 92px !important;
                    font-size: 18px !important;
                    padding: 15px 17px !important;
                }

                .home-menu-description {
                    font-size: 12px;
                    padding-left: 56px;
                    top: -47px;
                    margin-bottom: -28px;
                }

                .home-menu-description::before {
                    width: 41px;
                    height: 41px;
                    left: 12px;
                    top: -4px;
                }
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="home-screen">', unsafe_allow_html=True)

        # 상단 헤더
        st.markdown(
            """
            <div class="home-header">
                <div class="home-header-left">
                    <span class="home-header-house">⌂</span>
                    <span>주택관리공단 대구경북지사</span>
                </div>
                <div class="home-header-right">ⓘ 이용안내</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 중앙 제목
        st.markdown(
            """
            <div class="home-hero">
                <div class="home-hero-title">
                    🤖 기술업무 <span class="ai-blue">AI</span> 챗봇
                </div>
                <div class="home-hero-subtitle">
                    기술규정 검색부터 안내문 작성까지, AI가 빠르고 정확하게 도와드립니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="home-panel-wrap">', unsafe_allow_html=True)

        col_left, col_right = st.columns(2, gap="small")

        # -------------------------------------------------
        # 왼쪽 : 업무지원항목
        # -------------------------------------------------
        with col_left:

            st.markdown('<div class="home-blue-marker"></div>', unsafe_allow_html=True)

            st.markdown(
                """
                <div class="home-panel-head">
                    <div class="home-panel-icon home-blue-icon">📝</div>
                    <div>
                        <div class="home-panel-title home-blue-title">업무지원항목</div>
                        <div class="home-panel-desc">다양한 업무를 AI가 지원합니다.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "📄  안내문 작성                                      ›",
                key="home_notice_menu",
                use_container_width=True
            ):
                st.session_state.show_notice_generator = True
                st.rerun()

            st.markdown(
                """
                <div class="home-menu-description">
                    입력한 내용을 바탕으로 안내문을 작성합니다.
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "💬  기타 업무지원                                  ›",
                key="home_other_support",
                use_container_width=True
            ):
                st.info("기타 업무지원 기능은 준비 중입니다.")

            st.markdown(
                """
                <div class="home-menu-description">
                    추가 업무지원 기능을 준비하고 있습니다.
                </div>
                """,
                unsafe_allow_html=True
            )

        # -------------------------------------------------
        # 오른쪽 : 규정집 선택
        # -------------------------------------------------
        with col_right:

            st.markdown('<div class="home-green-marker"></div>', unsafe_allow_html=True)

            st.markdown(
                """
                <div class="home-panel-head">
                    <div class="home-panel-icon home-green-icon">📖</div>
                    <div>
                        <div class="home-panel-title home-green-title">규정집 선택</div>
                        <div class="home-panel-desc">찾고자 하는 규정집을 선택해주세요.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "🏢  시설업무                                        ›",
                key="rule_category_facility",
                use_container_width=True
            ):
                st.session_state.rulebook_category = "시설업무"
                st.rerun()

            st.markdown(
                """
                <div class="home-menu-description home-green-description">
                    시설·기계·전기 등 기술 관련<br>규정집을 확인합니다.
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "📋  행정업무                                        ›",
                key="rule_category_admin",
                use_container_width=True
            ):
                st.session_state.rulebook_category = "행정업무"
                st.rerun()

            st.markdown(
                """
                <div class="home-menu-description home-green-description">
                    행정·인사·예산·평가 등<br>행정 관련 규정집을 확인합니다.
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown('</div></div>', unsafe_allow_html=True)

    else:

        # -------------------------------------------------
        # 규정집 선택 화면
        # -------------------------------------------------

        category = st.session_state.rulebook_category

        st.markdown(
            '<div class="ai-avatar" style="font-size:55px;">📚</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="ai-greeting" style="font-size:21px;">'
            f'{category} 규정집 선택'
            '</div>',
            unsafe_allow_html=True
        )

        # 기존 첫 화면의 규정 안내 문구를 이곳으로 이동
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

        # -------------------------------------------------
        # 규정집 목록
        # -------------------------------------------------

        st.markdown(
            f'<div class="section-title">📄 {category} 규정집</div>',
            unsafe_allow_html=True
        )

        # -------------------------------------------------
        # 분야별 분류
        # -------------------------------------------------
        # 행정업무에는 "내부경영평가 편람"만 표시하고,
        # 나머지 모든 규정집은 시설업무에 표시합니다.

        admin_files = [
            filename for filename in filenames
            if "내부경영평가 편람" in filename
        ]

        facility_files = [
            filename for filename in filenames
            if filename not in admin_files
        ]

        category_files = (
            facility_files
            if category == "시설업무"
            else admin_files
        )

        if category_files:
            for i, filename in enumerate(category_files):
                if st.button(
                    f"📄  {filename}",
                    key=f"{category}_pdf_{i}",
                    use_container_width=True
                ):
                    st.session_state.selected_file = filename
                    st.session_state.messages = []
                    st.rerun()
        else:
            if category == "행정업무":
                st.info("행정업무 규정집이 없습니다.")
            else:
                st.info("시설업무 규정집이 없습니다.")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        if st.button(
            "↩️ 업무 분야 다시 선택",
            key="back_to_rule_categories",
            use_container_width=True
        ):
            st.session_state.rulebook_category = None
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
