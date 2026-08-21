from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import glob
import pickle


# ==========================================
# 설정
# ==========================================

PDF_FOLDER = "규정집"
VECTOR_FOLDER = "vector_db"

MODEL_NAME = "intfloat/multilingual-e5-small"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# ==========================================
# 폴더 생성
# ==========================================

os.makedirs(
    VECTOR_FOLDER,
    exist_ok=True
)


# ==========================================
# PDF 찾기
# ==========================================

pdf_files = glob.glob(
    os.path.join(
        PDF_FOLDER,
        "**",
        "*.pdf"
    ),
    recursive=True
)

pdf_files = list(
    dict.fromkeys(pdf_files)
)


if not pdf_files:

    print(
        "❌ 규정집 폴더에 PDF 파일이 없습니다."
    )

    print(
        f"PDF 파일을 '{PDF_FOLDER}' 폴더에 넣어주세요."
    )

    exit()


print("=" * 60)

print(
    f"📚 발견된 PDF : {len(pdf_files)}개"
)

for file in pdf_files:

    print(
        " -",
        file
    )

print("=" * 60)


# ==========================================
# PDF → 문서 조각
# ==========================================

documents = []


for file_path in pdf_files:

    filename = os.path.basename(
        file_path
    )

    print()
    print(
        f"📖 처리 중 : {filename}"
    )

    try:

        reader = PdfReader(
            file_path
        )

        total_pages = len(
            reader.pages
        )

        print(
            f"   총 {total_pages}페이지"
        )


        for page_num, page in enumerate(
            reader.pages,
            start=1
        ):

            text = page.extract_text()


            if not text:

                continue


            # ----------------------------------
            # 텍스트 정리
            # ----------------------------------

            text = text.replace(
                "\x00",
                " "
            )

            lines = []

            for line in text.splitlines():

                line = line.strip()

                if line:

                    lines.append(
                        line
                    )


            text = "\n".join(
                lines
            )


            if not text:

                continue


            # ----------------------------------
            # 페이지 내용을 여러 조각으로 분할
            # ----------------------------------

            start = 0

            while start < len(text):

                end = (
                    start
                    + CHUNK_SIZE
                )

                chunk_text = text[
                    start:end
                ].strip()


                if chunk_text:

                    documents.append(
                        {
                            "text": chunk_text,
                            "filename": filename,
                            "page": page_num,
                            "file_path": file_path
                        }
                    )


                start += (
                    CHUNK_SIZE
                    - CHUNK_OVERLAP
                )


    except Exception as e:

        print(
            f"   ❌ 오류 : {e}"
        )


print()
print("=" * 60)

print(
    f"📝 생성된 문서 조각 : {len(documents)}개"
)

print("=" * 60)


if not documents:

    print(
        "❌ PDF에서 텍스트를 추출하지 못했습니다."
    )

    print(
        "스캔 이미지 PDF라면 OCR 처리가 필요합니다."
    )

    exit()


# ==========================================
# 임베딩 모델
# ==========================================

print()
print(
    "🧠 임베딩 모델을 불러오는 중..."
)

model = SentenceTransformer(
    MODEL_NAME
)


# ==========================================
# 벡터 생성
# ==========================================

print(
    "🔄 문서 벡터화 시작..."
)

texts = [
    doc["text"]
    for doc in documents
]


passages = [
    "passage: " + text
    for text in texts
]


embeddings = model.encode(
    passages,
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=16
)


embeddings = np.asarray(
    embeddings,
    dtype="float32"
)


# ==========================================
# FAISS DB 생성
# ==========================================

dimension = embeddings.shape[1]


index = faiss.IndexFlatIP(
    dimension
)


index.add(
    embeddings
)


# ==========================================
# 저장
# ==========================================

index_path = os.path.join(
    VECTOR_FOLDER,
    "index.faiss"
)


documents_path = os.path.join(
    VECTOR_FOLDER,
    "documents.pkl"
)


faiss.write_index(
    index,
    index_path
)


with open(
    documents_path,
    "wb"
) as f:

    pickle.dump(
        documents,
        f
    )


# ==========================================
# 완료
# ==========================================

print()
print("=" * 60)

print(
    "✅ 벡터 DB 생성 완료!"
)

print("=" * 60)

print(
    f"📁 {index_path}"
)

print(
    f"📁 {documents_path}"
)

print(
    f"📚 PDF : {len(pdf_files)}개"
)

print(
    f"📝 문서 조각 : {len(documents)}개"
)

print()
print(
    "이제 index.faiss와 documents.pkl을"
)

print(
    "GitHub의 vector_db 폴더에 업로드하세요."
)

print("=" * 60)
