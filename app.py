import streamlit as st
from google import genai

import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer

import os
import re
import xml.etree.ElementTree as ET
import time
import zipfile
import tempfile
import shutil
from pathlib import Path
from html import escape
from urllib.request import urlopen, Request
from urllib.parse import quote


# =========================================================
# 1. 페이지 설정
# =========================================================

st.set_page_config(
    page_title="주택관리공단 대구경북지사",
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

    .stApp {
        background: #f5f8fc;
    }

    .main .block-container {
        max-width: 720px;
        padding-top: 18px;
        padding-left: 18px;
        padding-right: 18px;
        padding-bottom: 80px;
    }

    footer {
        visibility: hidden;
    }

    [data-testid="stToolbar"] {
        visibility: hidden;
        height: 0;
    }

    .top-title {
        font-size: 30px;
        font-weight: 800;
        color: #16283d;
        margin-top: 0px;
        margin-bottom: 5px;
        letter-spacing: -1.5px;
    }

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

        .notice-preview {
        line-height: 2.0;
        white-space: pre-line;
        font-size: 14px;
        color: #334b64;
    }

.welcome-text {
        color: #66758a;
        font-size: 13px;
        line-height: 1.7;
    }

    .section-title {
        color: #24364d;
        font-size: 17px;
        font-weight: 800;
        margin-top: 18px;
        margin-bottom: 10px;
    }

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

    [data-testid="stChatMessage"] {
        border-radius: 18px;
        margin-bottom: 10px;
    }

    [data-testid="stChatMessage"]:has(
        [data-testid="chatAvatarIcon-user"]
    ) {
        background-color: #eaf4ff;
    }

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

    hr {
        border: none;
        border-top: 1px solid #dce5ef;
        margin-top: 18px;
        margin-bottom: 18px;
    }

    @media (max-width: 600px) {

        .main .block-container {
            padding-left: 14px;
            padding-right: 14px;
            padding-top: 10px;
            padding-bottom: 70px;
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
# 4. 상단 제목
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
    st.error("Gemini API 키가 설정되지 않았습니다.")
    st.info(
        "Streamlit Cloud → Manage app → Settings → Secrets에서 "
        "GEMINI_API_KEY를 확인해주세요."
    )
    st.stop()


try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"Gemini 클라이언트 생성 오류: {e}")
    st.stop()


# =========================================================
# 6. 안내문 HWPX 양식 주소
# =========================================================
# GitHub에 올린 실제 파일 주소로 변경하세요.
#
# 예:
# https://raw.githubusercontent.com/아이디/저장소/main/templates/붙임.소독 안내문.hwpx
#
# 파일명을 URL에 넣을 때 한글/공백은 urllib이 처리할 수 있도록
# 아래처럼 raw.githubusercontent.com 주소를 사용합니다.
# =========================================================

# =========================================================
# HWPX 양식 설정
# =========================================================
# GitHub templates 폴더:
# templates/notice_template.hwpx
#
# 아래 3곳만 본인 GitHub 정보로 수정하세요.
# =========================================================

GITHUB_USERNAME = "Fin1041"
GITHUB_REPOSITORY = "AI"
GITHUB_BRANCH = "main"
HWPX_FILENAME = "notice_template.hwpx"

HWPX_TEMPLATE_URL = (
    f"https://raw.githubusercontent.com/"
    f"{GITHUB_USERNAME}/"
    f"{GITHUB_REPOSITORY}/"
    f"{GITHUB_BRANCH}/"
    f"templates/{HWPX_FILENAME}"
)

# =========================================================
# 7. 임베딩 모델
# =========================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "intfloat/multilingual-e5-small"
    )


with st.spinner("🔎 검색 시스템을 준비하고 있습니다..."):
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


if not os.path.exists(INDEX_PATH):
    st.error("❌ vector_db/index.faiss 파일이 없습니다.")
    st.stop()

if not os.path.exists(DOCUMENTS_PATH):
    st.error("❌ vector_db/documents.pkl 파일이 없습니다.")
    st.stop()


@st.cache_resource
def load_vector_database():

    index = faiss.read_index(INDEX_PATH)

    with open(
        DOCUMENTS_PATH,
        "rb"
    ) as f:
        documents = pickle.load(f)

    return index, documents


with st.spinner("📚 규정집 검색 DB를 불러오는 중..."):

    vector_index, documents = (
        load_vector_database()
    )


# =========================================================
# 9. PDF 목록
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
# 10. 검색 함수
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

    query_embedding = embedding_model.encode(
        ["query: " + query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

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
# 11. HWPX 다운로드
# =========================================================

def download_hwpx_template(url):

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    with urlopen(request, timeout=30) as response:
        data = response.read()

    if not data:
        raise RuntimeError(
            "GitHub에서 HWPX 파일을 받지 못했습니다."
        )

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".hwpx"
    )

    try:
        temp_file.write(data)
        temp_file.flush()
    finally:
        temp_file.close()

    return temp_file.name


# =========================================================
# 12. HWPX XML 치환
# =========================================================

def create_hwpx(
    template_path,
    output_path,
    subject,
    date,
    company,
    phone,
    office,
    notice_content
):

    # 사용자가 제공한 실제 HWPX 양식의 입력 위치
    replacements = {
        "{{건 명}}": subject,
        "{{건명}}": subject,
        "{{일 시}}": date,
        "{{일시}}": date,
        "{{업 체}}": company,
        "{{업체}}": company,
        "{{전화번호}}": phone,
        "{{관리소명}}": office,
        "{{안내내용}}": notice_content,
    }

    with tempfile.TemporaryDirectory() as temp_dir:

        with zipfile.ZipFile(
            template_path,
            "r"
        ) as zip_ref:
            zip_ref.extractall(temp_dir)

        replaced_count = 0

        for root, dirs, files in os.walk(temp_dir):

            for filename in files:

                if not filename.lower().endswith(".xml"):
                    continue

                file_path = os.path.join(
                    root,
                    filename
                )

                try:
                    with open(
                        file_path,
                        "r",
                        encoding="utf-8"
                    ) as f:
                        xml_text = f.read()
                except (UnicodeDecodeError, OSError):
                    continue

                original_text = xml_text

                for key, value in replacements.items():

                    # XML 특수문자 안전 처리
                    safe_value = escape(
                        str(value),
                        quote=False
                    )

                    count = xml_text.count(key)

                    if count:
                        xml_text = xml_text.replace(
                            key,
                            safe_value
                        )
                        replaced_count += count

                if xml_text != original_text:
                    with open(
                        file_path,
                        "w",
                        encoding="utf-8"
                    ) as f:
                        f.write(xml_text)

        if replaced_count == 0:
            raise RuntimeError(
                "HWPX 양식에서 {{건 명}}, {{일 시}}, "
                "{{업 체}}, {{전화번호}}, {{관리소명}} "
                "치환 위치를 찾지 못했습니다."
            )

        # HWPX는 ZIP 기반이므로 다시 압축
        with zipfile.ZipFile(
            output_path,
            "w",
            compression=zipfile.ZIP_DEFLATED
        ) as zip_ref:

            for root, dirs, files in os.walk(temp_dir):

                for filename in files:

                    file_path = os.path.join(
                        root,
                        filename
                    )

                    arcname = os.path.relpath(
                        file_path,
                        temp_dir
                    )

                    zip_ref.write(
                        file_path,
                        arcname
                    )

    return replaced_count


# =========================================================
# 13. Gemini 안내문 문구 생성
# =========================================================



def generate_content_with_retry(
    prompt,
    max_attempts=5
):
    """
    Gemini 503/UNAVAILABLE 일시 오류를 자동 재시도한다.
    503이 아닌 오류는 즉시 호출자에게 전달한다.
    """
    last_error = None

    for attempt in range(max_attempts):

        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt
            )

            text = (response.text or "").strip()

            if not text:
                raise RuntimeError(
                    "Gemini가 빈 응답을 반환했습니다."
                )

            return text

        except Exception as e:

            last_error = e
            error_text = str(e).upper()

            is_temporary = (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
                or "429" in error_text
            )

            if not is_temporary:
                raise

            if attempt < max_attempts - 1:
                # 1, 2, 4, 8초 + 약간의 랜덤 지연
                import random
                delay = (2 ** attempt) + random.uniform(0.2, 1.0)
                time.sleep(delay)

    raise RuntimeError(
        "Gemini 서버가 일시적으로 혼잡하여 "
        f"{max_attempts}회 재시도했지만 응답하지 않았습니다. "
        "잠시 후 다시 생성해 주세요.\n\n"
        f"마지막 오류: {last_error}"
    )

def generate_notice_content(
    request_text,
    subject,
    date,
    company,
    phone,
    office,
    regulation_context=""
):
    """
    규정집 검색 결과를 우선 근거로 사용하여
    안내문 제목과 본문을 생성한다.

    본문:
    - 최대 5줄
    - 일시/업체명/전화번호/관리소명 제외
    - 건명/작업 목적/입주민 협조사항 중심
    - 확인되지 않은 법조문/의무사항 금지
    """

    prompt = f"""
너는 공동주택 관리사무소의 '안내문 작성 담당 AI'이다.

사용자가 입력한 건명과 요청사항을 바탕으로
입주민에게 게시할 공식 안내문 제목과 본문을 작성한다.

[기본 정보]
건명: {subject}

일시: {date}
업체: {company}
전화번호: {phone}
관리소명: {office}

※ 일시, 업체, 전화번호, 관리소명은 한글 양식의 다른 항목에 자동으로 들어가므로
안내내용 본문에는 절대로 작성하지 않는다.

[사용자 요청]
{request_text}

[관련 규정집/문서 검색 결과]
{regulation_context if regulation_context else "관련 규정집에서 관련 내용을 찾지 못했거나 현재 규정집을 선택하지 않았습니다."}

[가장 중요한 작성 원칙]
1. '규정집/공식 문서 검색 결과'에 있는 내용을 최우선 근거로 사용한다.
2. 검색 결과에 명시된 법령, 규정, 관리기준, 절차가 있으면 그 내용을
   왜 해당 작업을 하는지 설명하는 근거로 자연스럽게 반영한다.
3. 검색 결과에 없는 법조문 번호나 의무사항을 절대로 만들어내지 않는다.
4. 일반 상식이나 인터넷에서 알고 있는 내용을 근거가 있는 것처럼 쓰지 않는다.
5. 규정집에 근거가 충분하지 않으면 특정 조문을 억지로 작성하지 말고
   '관련 유지관리 기준에 따라'와 같이 확인 가능한 수준으로 표현한다.
6. 제목은 '건명 + 실시/시행/점검/작업 안내' 형태를 우선 사용한다.
7. 본문에는 반드시 건명과 관련된 핵심 내용이 포함되어야 한다.
8. 본문에는 '왜 실시하는지', '어떤 기준에 따라 관리하는지',
   '입주민에게 필요한 협조사항'을 중심으로 작성한다.
9. 일시, 날짜, 시간, 업체명, 전화번호, 관리소명은 본문에서 제외한다.
10. 본문은 최대 5줄이다.
11. 한 줄에는 한 문장만 작성한다.
12. 문장은 짧고 명확한 행정 안내문체로 작성한다.
13. 불필요한 인사말, 장황한 설명, AI라는 표현은 쓰지 않는다.
14. 과장된 표현이나 법적 의무를 단정하지 않는다.

[출력 형식]
[제목]
제목 한 줄

[본문]
문장 1
문장 2
문장 3
문장 4
문장 5

[근거]
실제로 검색 결과에서 확인된 규정/기준만 1~2개 간단히 작성한다.
검색 결과에서 확인되지 않으면 '검색된 규정집에서 특정 조문 확인 안 됨'이라고 작성한다.
"""

    text = generate_content_with_retry(
        prompt=prompt,
        max_attempts=5
    )

    title = ""
    body = ""
    basis = ""

    if "[제목]" in text:
        after_title = text.split("[제목]", 1)[1]

        if "[본문]" in after_title:
            title, after_body = after_title.split(
                "[본문]", 1
            )
        else:
            title = after_title
            after_body = ""

        if "[근거]" in after_body:
            body, basis = after_body.split(
                "[근거]", 1
            )
        else:
            body = after_body

    title = title.strip() or f"{subject} 안내"
    body = body.strip()
    basis = basis.strip()

    # AI가 번호/불릿을 붙였을 경우 제거
    body = re.sub(
        r"^\s*(?:[-•·]|\d+[\.\)])\s*",
        "",
        body,
        flags=re.MULTILINE
    )

    # 빈 줄 제거
    lines = [
        line.strip()
        for line in body.splitlines()
        if line.strip()
    ]

    # AI가 줄바꿈 없이 반환한 경우 문장 단위로 분리
    if len(lines) == 1:
        one = lines[0]

        # 일반적인 한국어 문장 종결 뒤에서 분리
        one = re.sub(
            r"([다요함됨임])\.\s+",
            r"\1.\n",
            one
        )

        lines = [
            x.strip()
            for x in one.splitlines()
            if x.strip()
        ]

    # 본문에 들어가면 안 되는 정보 제거
    forbidden_patterns = [
        date,
        company,
        phone,
        office,
    ]

    cleaned = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # 입력값 전체가 들어간 경우 제거
        if any(
            p and p.strip() in line
            for p in forbidden_patterns
        ):
            continue

        cleaned.append(line)

    # 최대 5줄
    body = "\n".join(cleaned[:5])

    return title, body, basis


def _replace_first_text_node(paragraph_xml, value):
    """
    기존 hp:p / hp:run 구조는 유지하고 첫 hp:t 텍스트만 변경한다.
    빈 문단인 경우 기존 hp:run의 속성을 유지한 채 hp:t만 추가한다.
    """
    safe = escape(str(value), quote=False)

    t_match = re.search(
        r"<hp:t\b[^>]*>.*?</hp:t>",
        paragraph_xml,
        re.DOTALL
    )

    if t_match:
        return (
            paragraph_xml[:t_match.start()]
            + "<hp:t>"
            + safe
            + "</hp:t>"
            + paragraph_xml[t_match.end():]
        )

    # 빈 문단의 self-closing hp:run을 확장
    run_match = re.search(
        r"<hp:run\b([^>]*)/>",
        paragraph_xml
    )

    if run_match:
        attrs = run_match.group(1)

        run_xml = (
            "<hp:run"
            + attrs
            + "><hp:t>"
            + safe
            + "</hp:t></hp:run>"
        )

        return (
            paragraph_xml[:run_match.start()]
            + run_xml
            + paragraph_xml[run_match.end():]
        )

    raise RuntimeError(
        "본문용 기존 HWPX 문단에서 텍스트를 넣을 위치를 찾지 못했습니다."
    )


def create_hwpx_safely(
    template_path,
    output_path,
    replacements,
    notice_content
):
    """
    원본 HWPX 구조를 유지하면서 section0.xml의 기존 텍스트만 교체한다.

    중요:
    - 새 namespace/tag를 생성하지 않는다.
    - 새 문단을 임의 생성하지 않는다.
    - 템플릿에 미리 만들어진 본문 문단 5개를 사용한다.
    - mimetype은 ZIP 첫 항목/비압축으로 유지한다.
    """

    lines = [
        line.strip()
        for line in str(notice_content).splitlines()
        if line.strip()
    ][:5]

    if not lines:
        raise RuntimeError(
            "AI가 생성한 안내내용이 비어 있습니다."
        )

    with zipfile.ZipFile(template_path, "r") as zin:

        names = zin.namelist()

        if not names:
            raise RuntimeError("HWPX 파일이 비어 있습니다.")

        if names[0] != "mimetype":
            raise RuntimeError(
                "HWPX 템플릿의 mimetype 위치가 올바르지 않습니다."
            )

        section_name = "Contents/section0.xml"

        if section_name not in names:
            raise RuntimeError(
                "Contents/section0.xml을 찾을 수 없습니다."
            )

        data = {
            name: zin.read(name)
            for name in names
        }

        xml = data[section_name].decode("utf-8")

        # -------------------------------------------------
        # 일반 placeholder 치환
        # -------------------------------------------------
        for key, value in replacements.items():

            if key == "{{안내내용}}":
                continue

            xml = xml.replace(
                key,
                escape(str(value), quote=False)
            )

        # -------------------------------------------------
        # {{안내내용}}이 들어있는 기존 문단부터
        # 본문용 5개 기존 문단을 순서대로 찾는다.
        # -------------------------------------------------
        paragraphs = list(
            re.finditer(
                r"<hp:p\b[^>]*>.*?</hp:p>",
                xml,
                re.DOTALL
            )
        )

        body_index = None

        for i, match in enumerate(paragraphs):

            if "{{안내내용}}" in match.group(0):
                body_index = i
                break

        if body_index is None:
            raise RuntimeError(
                "HWPX 템플릿에서 {{안내내용}}을 찾지 못했습니다."
            )

        targets = paragraphs[
            body_index:body_index + 5
        ]

        if len(targets) < 5:
            raise RuntimeError(
                "HWPX 템플릿에 본문용 기존 문단 5개가 없습니다."
            )

        # -------------------------------------------------
        # 기존 5개 문단을 그대로 사용하고 텍스트만 교체
        # -------------------------------------------------
        pieces = []
        cursor = 0

        for i, match in enumerate(targets):

            pieces.append(
                xml[cursor:match.start()]
            )

            paragraph = match.group(0)

            value = (
                lines[i]
                if i < len(lines)
                else ""
            )

            paragraph = _replace_first_text_node(
                paragraph,
                value
            )

            pieces.append(paragraph)
            cursor = match.end()

        pieces.append(
            xml[cursor:]
        )

        xml = "".join(pieces)

        # XML 파싱 검사
        try:
            ET.fromstring(xml)
        except ET.ParseError as e:
            raise RuntimeError(
                "수정된 HWPX XML이 올바르지 않습니다: "
                + str(e)
            ) from e

        data[section_name] = xml.encode("utf-8")

        # -------------------------------------------------
        # 원본 ZIP 메타데이터 최대한 유지
        # -------------------------------------------------
        with zipfile.ZipFile(
            output_path,
            "w"
        ) as zout:

            for name in names:

                original_info = zin.getinfo(name)
                info = copy.copy(original_info)

                if name == "mimetype":
                    info.compress_type = zipfile.ZIP_STORED

                zout.writestr(
                    info,
                    data[name]
                )

    # -----------------------------------------------------
    # 최종 결과 HWPX 자체 검사
    # -----------------------------------------------------
    with zipfile.ZipFile(
        output_path,
        "r"
    ) as check_zip:

        bad = check_zip.testzip()

        if bad is not None:
            raise RuntimeError(
                "생성된 HWPX ZIP 검증 실패: "
                + bad
            )

        names_after = check_zip.namelist()

        if not names_after or names_after[0] != "mimetype":
            raise RuntimeError(
                "생성된 HWPX의 mimetype 위치가 잘못되었습니다."
            )

        if check_zip.read("mimetype") != b"application/hwp+zip":
            raise RuntimeError(
                "생성된 HWPX의 mimetype 값이 잘못되었습니다."
            )

    return output_path

# =========================================================
# 14. 안내문 생성 화면
# =========================================================

if st.session_state.show_notice_generator:

    st.markdown(
        '<div class="ai-avatar" style="font-size:55px;">📄</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="ai-greeting" style="font-size:21px;">'
        'AI 안내문 생성'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="welcome-card">

        <div class="welcome-title">
        ✨ 안내문 내용을 AI가 작성합니다
        </div>

        <div class="welcome-text">
        안내문 요청내용을 입력하면 AI가 관련 법령과 공식 기준을
        확인할 수 있는 범위에서 검토하여 5줄 이내의 문구를 작성합니다.
        확인되지 않은 법적 의무는 임의로 작성하지 않습니다.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    notice_date = st.text_input(
        "① 공고일자",
        placeholder="예: 2026년 9월 1일"
    )

    notice_deadline = st.text_input(
        "② 공고기한",
        placeholder="예: 2026년 9월 10일까지"
    )

    subject = st.text_input(
        "③ 건명",
        placeholder="예: 보일러 세관"
    )

    date = st.text_input(
        "④ 일시",
        placeholder="예: 2026년 9월 10일 09:00~17:00"
    )

    company = st.text_input(
        "⑤ 업체",
        placeholder="예: ○○설비"
    )

    phone = st.text_input(
        "⑥ 전화번호",
        placeholder="예: 053-123-4567"
    )

    office = st.text_input(
        "⑦ 관리소명",
        placeholder="예: ○○관리소"
    )

    request_text = st.text_area(
        "⑧ 안내문 내용 요청",
        placeholder=(
            "예: 보일러 세관에 대해서 관련 법규나 근거를 "
            "바탕으로 입주민 안내문을 5줄 이내로 만들어줘"
        ),
        height=120
    )

    st.markdown("---")

    if st.button(
        "✨ AI 안내문 생성",
        key="create_notice",
        use_container_width=True
    ):

        missing = []

        if not notice_date.strip():
            missing.append("공고일자")
        if not subject.strip():
            missing.append("건명")
        if not date.strip():
            missing.append("일시")
        if not company.strip():
            missing.append("업체")
        if not phone.strip():
            missing.append("전화번호")
        if not office.strip():
            missing.append("관리소명")
        if not request_text.strip():
            missing.append("안내문 내용 요청")

        if missing:

            st.warning(
                "다음 항목을 입력해주세요: "
                + ", ".join(missing)
            )

        else:

            try:

                with st.spinner(
                    "🤖 관련 규정을 확인하고 안내문을 작성하고 있습니다..."
                ):

                    # 선택된 규정집에서 건명/요청사항과 관련된 내용을 검색
                    selected_doc = st.session_state.get(
                        "selected_file"
                    )

                    # 규정집 선택 화면에서 안내문 생성 버튼을 눌렀거나,
                    # 규정집 선택 후 안내문 생성 버튼을 눌렀을 때
                    # 현재 선택된 PDF 이름을 사용한다.
                    regulation_results = []

                    if selected_doc:
                        regulation_results = search_documents(
                            query=f"{subject} {request_text}",
                            selected_filename=selected_doc,
                            top_k=8
                        )

                    regulation_context_parts = []

                    for result in regulation_results:
                        regulation_context_parts.append(
                            f"[문서명] {result.get('filename', '')}\n"
                            f"[페이지] {result.get('page', '')}\n"
                            f"[내용]\n{result.get('text', '')}"
                        )

                    regulation_context = "\n\n".join(
                        regulation_context_parts
                    )

                    title, notice_content, basis = (
                        generate_notice_content(
                            request_text=request_text,
                            subject=subject,
                            date=date,
                            company=company,
                            phone=phone,
                            office=office,
                            regulation_context=regulation_context
                        )
                    )

                st.success("✅ AI 안내문 작성이 완료되었습니다.")

                st.markdown(
                    '<div class="section-title">📝 생성 결과</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class="welcome-card">
                    <div class="welcome-title">
                    제목
                    </div>
                    <div style="font-size:17px;font-weight:800;
                    color:#243c56;margin-bottom:15px;">
                    {escape(title)}
                    </div>

                    <div class="welcome-title">
                    안내내용
                    </div>
                    <div class="notice-preview">
                    {escape(notice_content).replace(chr(10), "<br>")}
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if basis:
                    st.info(
                        "📚 AI가 참고한 근거\n\n"
                        + basis
                    )

                with st.spinner(
                    "📄 한글파일 양식에 내용을 넣고 있습니다..."
                ):

                    template_path = download_hwpx_template(
                        HWPX_TEMPLATE_URL
                    )

                    output_path = os.path.join(
                        tempfile.gettempdir(),
                        "안내문_완성본.hwpx"
                    )

                    replacements = {
                        "{{공고일자}}": notice_date,
                        "{{공고기한}}": notice_deadline,
                        "{{제목}}": title,
                        "{{건 명}}": subject,
                        "{{건명}}": subject,
                        "{{일 시}}": date,
                        "{{일시}}": date,
                        "{{업 체}}": company,
                        "{{업체}}": company,
                        "{{전화번호}}": phone,
                        "{{관리소명}}": office,
                    }

                    # 원본 HWPX의 ZIP 구조를 유지한 채
                    # Contents/section0.xml만 안전하게 수정한다.
                    create_hwpx_safely(
                        template_path=template_path,
                        output_path=output_path,
                        replacements=replacements,
                        notice_content=notice_content
                    )

                with open(
                    output_path,
                    "rb"
                ) as f:
                    hwpx_data = f.read()

                st.download_button(
                    label="📥 완성된 안내문 한글파일 다운로드",
                    data=hwpx_data,
                    file_name=f"{subject}_안내문.hwpx",
                    mime="application/octet-stream",
                    use_container_width=True
                )

            except Exception as e:

                error_text = str(e)
                upper_error = error_text.upper()

                if (
                    "503" in upper_error
                    or "UNAVAILABLE" in upper_error
                    or "RESOURCE_EXHAUSTED" in upper_error
                    or "429" in upper_error
                ):
                    st.warning(
                        "🤖 Gemini AI 서버가 현재 혼잡합니다.\n\n"
                        "자동으로 여러 차례 재시도했지만 응답하지 않았습니다. "
                        "잠시 후 다시 **AI 안내문 생성**을 눌러주세요."
                    )

                elif "HWPX" in upper_error or "안내내용" in error_text:
                    st.error(
                        "❌ 한글파일 양식 처리 중 오류가 발생했습니다."
                    )
                    st.code(
                        error_text,
                        language="text"
                    )
                    st.info(
                        "GitHub의 templates/notice_template.hwpx가 "
                        "원본 템플릿인지 확인해주세요."
                    )

                else:
                    st.error(
                        "❌ 안내문 생성 중 오류가 발생했습니다."
                    )
                    st.code(
                        error_text,
                        language="text"
                    )

    st.markdown("")

    if st.button(
        "↩️ 처음 화면으로 돌아가기",
        key="close_notice",
        use_container_width=True
    ):

        st.session_state.show_notice_generator = False
        st.rerun()

    st.stop()


# =========================================================
# 15. 첫 화면
# =========================================================

if st.session_state.selected_file is None:

    st.markdown(
        '<div class="ai-avatar">🤖</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="ai-greeting">'
        '대구경북지사 직원 여러분 안녕하세요^^<br>'
        '무엇을 도와드릴까요?'
        '</div>',
        unsafe_allow_html=True
    )

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
        key="notice_generator_home",
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
# 16. 규정집 선택 후 화면
# =========================================================

else:

    selected_file = st.session_state.selected_file


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


    st.markdown(
        f"""
        <div class="selected-document">

        <div class="selected-document-title">
        📚 현재 선택된 규정집
        </div>

        <div class="selected-document-name">
        {escape(selected_file)}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    if st.button(
        "↩️ 다른 규정집 선택",
        key="back_to_documents",
        use_container_width=True
    ):

        st.session_state.selected_file = None
        st.session_state.messages = []
        st.rerun()


    # 안내문 생성 버튼도 규정집 선택 후 사용할 수 있게 함
    if st.button(
        "📄 안내문 생성",
        key="notice_generator_selected",
        use_container_width=True
    ):

        st.session_state.show_notice_generator = True
        st.rerun()


    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    user_input = st.chat_input(
        "궁금한 내용을 입력하세요..."
    )


    if user_input:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )


        with st.chat_message("user"):
            st.markdown(user_input)


        with st.chat_message("assistant"):

            status = st.status(
                "📚 관련 규정을 검색하고 있습니다...",
                expanded=True
            )


            try:

                search_results = search_documents(
                    user_input,
                    st.session_state.selected_file,
                    top_k=6
                )


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

                    status.update(
                        label=(
                            f"관련 규정 "
                            f"{len(search_results)}건을 찾았습니다."
                        ),
                        state="running",
                        expanded=True
                    )


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


                    status.update(
                        label="✅ 답변 작성이 완료되었습니다.",
                        state="complete",
                        expanded=False
                    )


                    st.markdown(answer)


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


                st.markdown(error_answer)


                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_answer
                    }
                )
