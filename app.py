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
    /* 첫 화면 전용 디자인. 다른 화면의 기존 스타일은 유지 */
    .main .block-container:has(.home-screen) .top-title {
        background: linear-gradient(135deg, #182f49 0%, #203f60 100%);
        color: #ffffff !important;
        margin: -18px -18px 0 -18px;
        padding: 19px 28px;
        font-size: 20px;
        font-weight: 800;
        letter-spacing: -0.7px;
        border-radius: 0;
    }
    .main .block-container:has(.home-screen) hr { display:none; }

    .home-screen { width:100%; }
    .home-hero {
        text-align:center;
        padding:34px 8px 28px;
        background:#fff;
    }
    .home-hero-title {
        color:#173b68;
        font-size:44px;
        font-weight:900;
        letter-spacing:-2.5px;
        line-height:1.18;
    }
    .home-hero-title .ai-blue { color:#2b7ddd; }
    .home-hero-desc {
        margin-top:12px;
        color:#283a52;
        font-size:16px;
        letter-spacing:-0.6px;
    }

    .home-frame {
        border:2px solid #d7e1eb;
        border-radius:24px;
        padding:18px;
        background:#fff;
        box-sizing:border-box;
    }

    /* Streamlit column 자체를 이미지의 좌/우 색상 패널로 사용 */
    div[data-testid="column"]:has(.home-blue-anchor),
    div[data-testid="column"]:has(.home-green-anchor) {
        border-radius:21px;
        padding:27px 21px 25px !important;
        box-sizing:border-box;
    }
    div[data-testid="column"]:has(.home-blue-anchor) {
        background:linear-gradient(180deg,#edf6ff 0%,#e6f1fb 100%);
    }
    div[data-testid="column"]:has(.home-green-anchor) {
        background:linear-gradient(180deg,#eff9eb 0%,#e9f6e5 100%);
    }

    .home-anchor { display:none; }
    .home-panel-head {
        display:flex;
        align-items:center;
        gap:13px;
        margin-bottom:15px;
    }
    .home-circle {
        width:60px;
        height:60px;
        min-width:60px;
        border-radius:50%;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:33px;
    }
    .home-circle-blue { background:#2d82dc; }
    .home-circle-green { background:#23955f; }
    .home-panel-title {
        font-size:27px;
        font-weight:900;
        line-height:1.1;
        letter-spacing:-1.4px;
    }
    .home-panel-title.blue { color:#174d86; }
    .home-panel-title.green { color:#197043; }
    .home-panel-desc {
        margin-top:4px;
        color:#30445c;
        font-size:13px;
        line-height:1.4;
        letter-spacing:-0.4px;
    }

    /* 실제 클릭 버튼을 참고 이미지의 흰색 메뉴 카드로 */
    .home-screen .stButton { margin:0 0 6px 0; }
    .home-screen .stButton > button {
        min-height:48px !important;
        height:48px !important;
        padding:8px 15px !important;
        border-radius:15px !important;
        border:1px solid #d6e1eb !important;
        background:#ffffff !important;
        color:#1d3551 !important;
        box-shadow:0 3px 10px rgba(37,74,108,.07) !important;
        transform:none !important;
        text-align:left !important;
    }
    .home-screen .stButton > button:hover {
        background:#ffffff !important;
        transform:translateY(-1px) !important;
        box-shadow:0 5px 13px rgba(37,74,108,.11) !important;
    }
    .home-screen .stButton > button p {
        white-space:nowrap !important;
        line-height:1.2 !important;
        font-size:16px !important;
        font-weight:800 !important;
        color:#1d3551 !important;
        margin:0 !important;
    }
    .home-menu-desc {
        margin:-1px 7px 12px 7px;
        color:#30445c;
        font-size:12px;
        line-height:1.45;
        letter-spacing:-0.35px;
    }

    @media (max-width:760px) {
        .main .block-container:has(.home-screen) .top-title {
            margin-left:-14px;
            margin-right:-14px;
            padding-left:18px;
            padding-right:18px;
            font-size:17px;
        }
        .home-hero { padding:25px 4px 22px; }
        .home-hero-title { font-size:31px; letter-spacing:-1.6px; }
        .home-hero-desc { font-size:13px; line-height:1.5; }
        .home-frame { padding:10px; border-radius:18px; }
        div[data-testid="column"]:has(.home-blue-anchor),
        div[data-testid="column"]:has(.home-green-anchor) {
            padding:21px 15px 18px !important;
            margin-bottom:10px;
        }
        .home-circle { width:50px; height:50px; min-width:50px; font-size:27px; }
        .home-panel-title { font-size:22px; }
        .home-panel-desc { font-size:12px; }
        .home-screen .stButton > button { min-height:46px !important; height:46px !important; }
        .home-menu-desc { font-size:11px; margin-bottom:10px; }
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


def download_notice_template():
    data = _github_download(
        HWPX_TEMPLATE_URL
    )

    try:
        with zipfile.ZipFile(
            __import__("io").BytesIO(data),
            "r"
        ) as z:
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

            section = z.read(
                "Contents/section0.xml"
            ).decode("utf-8")

            # 새 양식은 반드시 안내내용1~5를 사용
            for i in range(1, 6):
                if f"{{{{안내내용{i}}}}}" not in section:
                    raise RuntimeError(
                        f"notice_template.hwpx에 "
                        f"{{{{안내내용{i}}}}}이 없습니다."
                    )

    except zipfile.BadZipFile as e:
        raise RuntimeError(
            "GitHub에서 받은 notice_template.hwpx가 손상되었습니다."
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

건명:
{subject}

요청:
{request_text}

목적:
안내문에 넣을 수 있는 실제 법령 또는 공식 기준 근거가 있는지 확인한다.

반드시 지켜라.
1. 대한민국 현행 법령만 대상으로 한다.
2. 공식 법령 출처는 국가법령정보센터(law.go.kr)를 우선한다.
3. 실제 확인하지 못한 법률명, 조문번호, 시행일은 절대 만들어내지 않는다.
4. 관련 근거가 확실하지 않으면 '근거 미확인'이라고 답한다.
5. 근거가 확인되면 법령명과 조문번호, 안내문에 사용할 수 있는 핵심 취지만 짧게 작성한다.
6. 안내문 본문에 넣을 수 있는 쉬운 한국어로 요약한다.

출력 형식:
[법령명]
...
[조문]
...
[핵심근거]
...
[확인상태]
확인 / 근거 미확인
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
너는 공동주택 관리사무소의 공식 안내문 작성 담당자이다.

[건명]
{subject}

[사용자 요청]
{request_text}

[공식 법령 근거 확인 결과]
{law_context if law_context else "확인된 법령 근거 없음"}

[작성 규칙]
1. 제목은 15자 이하.
2. 안내내용은 무조건 5줄 이내.
3. 각 줄은 짧고 읽기 쉽게 작성한다.
4. 한 줄은 공백 제외 약 20~25글자를 목표로 한다.
5. 건명 → 목적/필요성 → 법령근거 → 협조사항 순으로 작성한다.
6. 실제 확인된 법령 근거가 있으면 반드시 한 줄에 직접 포함한다.
7. 법령명과 조문번호는 확인된 경우에만 쓴다.
8. 확인되지 않은 조문번호는 절대 작성하지 않는다.
9. 일시·날짜·업체명·전화번호·관리소명은 본문에서 제외한다.
10. 어려운 법률 표현은 줄이고 입주민이 이해하기 쉽게 쓴다.
11. 같은 내용 반복 금지.
12. 본문은 반드시 5줄 이내다.

[출력]
[제목]
15자 이하

[본문]
문장1
문장2
문장3
문장4
문장5
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

    if len(custom_fields) != 5:
        raise RuntimeError("사용자 지정 항목은 5개가 필요합니다.")

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
            label, value = custom_fields[idx]

            safe_label = html.escape(str(label), quote=False)
            safe_value = html.escape(str(value), quote=False)

            # 같은 문단 안의 placeholder 두 개만 해당 순서대로 치환
            new_paragraph = paragraph.replace("{{항목명}}", safe_label, 1)
            new_paragraph = new_paragraph.replace("{{입력내용}}", safe_value, 1)

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

        custom_fields = [
            (str(field1_label).strip(), str(field1_value).strip()),
            (str(field2_label).strip(), str(field2_value).strip()),
            (str(field3_label).strip(), str(field3_value).strip()),
            (str(field4_label).strip(), str(field4_value).strip()),
            (str(field5_label).strip(), str(field5_value).strip()),
        ]

        missing = []
        if not notice_date.strip():
            missing.append("공고일자")
        if not notice_deadline.strip():
            missing.append("공고기한")

        for idx, (label, value) in enumerate(custom_fields, 1):
            if not label:
                missing.append(f"{idx}번 항목명")
            if not value:
                missing.append(f"{idx}번 입력내용")

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
                subject = custom_fields[0][1]
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

            st.markdown(
                f"""
                <div class="welcome-card">
                <div class="welcome-title">제목</div>
                <div style="font-size:18px;font-weight:800;
                color:#243c56;margin-bottom:18px;">
                {safe_title}
                </div>

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

        st.markdown(
            """
            <div class="home-screen">
                <div class="home-hero">
                    <div class="home-hero-title">🤖 기술업무 <span class="ai-blue">AI</span> 챗봇</div>
                    <div class="home-hero-desc">기술규정 검색부터 안내문 작성까지, AI가 빠르고 정확하게 도와드립니다.</div>
                </div>
                <div class="home-frame">
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2, gap="medium")

        with col1:
            st.markdown('<span class="home-anchor home-blue-anchor"></span>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="home-panel-head">
                    <div class="home-circle home-circle-blue">📝</div>
                    <div>
                        <div class="home-panel-title blue">업무지원항목</div>
                        <div class="home-panel-desc">다양한 업무를 AI가 지원합니다.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "📄  안내문 작성   ›",
                key="home_notice_menu",
                use_container_width=True
            ):
                st.session_state.show_notice_generator = True
                st.rerun()

            st.markdown(
                '<div class="home-menu-desc">입력한 내용을 바탕으로 안내문을 작성합니다.</div>',
                unsafe_allow_html=True
            )

            st.button(
                "💬  기타 업무지원   ›",
                key="home_other_support",
                use_container_width=True
            )

            st.markdown(
                '<div class="home-menu-desc">추가 업무지원 기능을 제공합니다.</div>',
                unsafe_allow_html=True
            )

        with col2:
            st.markdown('<span class="home-anchor home-green-anchor"></span>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="home-panel-head">
                    <div class="home-circle home-circle-green">📚</div>
                    <div>
                        <div class="home-panel-title green">규정집 선택</div>
                        <div class="home-panel-desc">찾고자 하는 규정집을 선택해주세요.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "🏢  시설업무   ›",
                key="rule_category_facility",
                use_container_width=True
            ):
                st.session_state.rulebook_category = "시설업무"
                st.rerun()

            st.markdown(
                '<div class="home-menu-desc">시설·기계·전기 등 기술 관련 규정집을 확인합니다.</div>',
                unsafe_allow_html=True
            )

            if st.button(
                "📋  행정업무   ›",
                key="rule_category_admin",
                use_container_width=True
            ):
                st.session_state.rulebook_category = "행정업무"
                st.rerun()

            st.markdown(
                '<div class="home-menu-desc">행정·인사·예산·평가 등 행정 관련 규정집을 확인합니다.</div>',
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
