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
import shutil
from pathlib import Path
from html import escape
from urllib.request import urlopen, Request


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

HWPX_TEMPLATE_URL = (
    "https://raw.githubusercontent.com/"
    "YOUR_GITHUB_USERNAME/"
    "YOUR_REPOSITORY/"
    "main/templates/붙임.소독%20안내문.hwpx"
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

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".hwpx"
    )

    temp_file.write(data)
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

    replacements = {
        "{{건 명}}": subject,
        "{{건명}}": subject,

        "{{일 시}}": date,
        "{{일시}}": date,

        "{{업 체}}": company,
        "{{업체}}": company,

        "{{전화번호}}": phone,
        "{{ 전화번호 }}": phone,

        "{{관리소명}}": office,
        "{{ 관리소명 }}": office,

        "{{안내내용}}": notice_content,
        "{{ 안내내용 }}": notice_content,
    }

    with tempfile.TemporaryDirectory() as temp_dir:

        with zipfile.ZipFile(
            template_path,
            "r"
        ) as zip_ref:

            zip_ref.extractall(temp_dir)


        replaced_count = 0

        # HWPX 내부 XML을 모두 검사
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

                except Exception:
                    continue


                original_text = xml_text


                for key, value in replacements.items():

                    value = str(value)

                    # HWPX XML 안에서 사용할 수 있도록
                    # 기본 XML 특수문자를 처리
                    safe_value = escape(
                        value,
                        quote=False
                    )

                    before = xml_text

                    xml_text = xml_text.replace(
                        key,
                        safe_value
                    )

                    if xml_text != before:
                        replaced_count += 1


                if xml_text != original_text:

                    with open(
                        file_path,
                        "w",
                        encoding="utf-8"
                    ) as f:

                        f.write(xml_text)


        # 다시 HWPX 압축
        with zipfile.ZipFile(
            output_path,
            "w",
            zipfile.ZIP_DEFLATED
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

def generate_notice_content(
    subject,
    date,
    company,
    phone,
    office
):

    prompt = f"""
너는 공동주택 관리사무소의 안내문 작성 담당자이다.

사용자가 입력한 정보를 이용하여
입주민에게 전달할 안내문 본문을 작성한다.

[입력 정보]

건명:
{subject}

일시:
{date}

업체:
{company}

전화번호:
{phone}

관리소명:
{office}

[작성 원칙]

1. 정중하고 공공기관 안내문에 적합한 문체를 사용한다.
2. 입주민이 쉽게 이해할 수 있도록 작성한다.
3. 입력된 정보에 없는 새로운 사실은 임의로 만들지 않는다.
4. 건명과 일시를 중심으로 자연스럽게 안내한다.
5. 기존 양식에 이미 들어 있는 법령 문구와 주의사항은
   별도로 반복하지 않는다.
6. 전화번호와 관리소명은 별도 양식에 들어가므로
   본문에서 반복하지 않는다.
7. 제목을 작성하지 않는다.
8. 번호 매기기나 별도 머리말 없이 본문만 작성한다.
9. 마지막에는 입주민의 양해와 협조를 부탁하는
   자연스러운 문장을 넣는다.
10. 한국어로 작성한다.

[안내문 본문]
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

    return response.text.strip()


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
        '안내문 생성'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="welcome-card">

        <div class="welcome-title">
        ✨ 안내문 정보를 입력해주세요
        </div>

        <div class="welcome-text">
        건명, 일시, 업체, 전화번호, 관리소명을 입력하면
        AI가 안내문 본문을 작성하고 기존 한글 양식에 넣어드립니다.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    subject = st.text_input(
        "① 건명",
        placeholder="예: 세대 소독 실시"
    )


    date = st.text_input(
        "② 일시",
        placeholder="예: 2026년 9월 3일 09:00~17:00"
    )


    company = st.text_input(
        "③ 업체",
        placeholder="예: ○○방역"
    )


    phone = st.text_input(
        "④ 전화번호",
        placeholder="예: 053-123-4567"
    )


    office = st.text_input(
        "⑤ 관리소명",
        placeholder="예: ○○관리소"
    )


    st.markdown("---")


    if st.button(
        "✨ 안내문 생성",
        key="create_notice",
        use_container_width=True
    ):

        missing = []

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


        if missing:

            st.warning(
                "다음 항목을 입력해주세요: "
                + ", ".join(missing)
            )

        else:

            try:

                # -------------------------------------
                # Gemini 안내문 생성
                # -------------------------------------

                with st.spinner(
                    "🤖 AI가 안내문 문구를 작성하고 있습니다..."
                ):

                    notice_content = generate_notice_content(
                        subject=subject,
                        date=date,
                        company=company,
                        phone=phone,
                        office=office
                    )


                # -------------------------------------
                # GitHub HWPX 양식 가져오기
                # -------------------------------------

                with st.spinner(
                    "📄 한글 양식을 불러오고 있습니다..."
                ):

                    template_path = (
                        download_hwpx_template(
                            HWPX_TEMPLATE_URL
                        )
                    )


                # -------------------------------------
                # 완성 파일 생성
                # -------------------------------------

                output_path = os.path.join(
                    tempfile.gettempdir(),
                    "안내문_완성본.hwpx"
                )


                replaced_count = create_hwpx(
                    template_path=template_path,
                    output_path=output_path,
                    subject=subject,
                    date=date,
                    company=company,
                    phone=phone,
                    office=office,
                    notice_content=notice_content
                )


                # -------------------------------------
                # 결과 확인
                # -------------------------------------

                if replaced_count == 0:

                    st.error(
                        "HWPX 양식에서 치환할 입력 항목을 찾지 못했습니다.\n\n"
                        "GitHub의 HWPX 양식이 현재 코드에서 지정한 "
                        "{{건 명}}, {{일 시}}, {{업 체}}, "
                        "{{전화번호}}, {{관리소명}} 형식인지 확인해주세요."
                    )

                else:

                    st.success(
                        "✅ 안내문이 완성되었습니다."
                    )


                    st.markdown(
                        '<div class="section-title">'
                        '📝 AI가 작성한 안내문'
                        '</div>',
                        unsafe_allow_html=True
                    )


                    st.markdown(
                        f"""
                        <div class="welcome-card">
                        {escape(notice_content).replace(chr(10), '<br>')}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                    # ---------------------------------
                    # 다운로드
                    # ---------------------------------

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

                st.error(
                    "❌ 안내문 생성 중 오류가 발생했습니다.\n\n"
                    + str(e)
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
        key="notice_generator",
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
