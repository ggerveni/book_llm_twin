import os
import glob
from dataclasses import dataclass
from typing import List, Dict

from pypdf import PdfReader
from zenml import step


@dataclass
class RawDocument:
    doc_id: str
    text: str
    path: str


def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


@step(enable_cache=False)
def load_documents(data_dir: str) -> List[Dict]:
    """
    Load all .pdf and .txt files from data_dir into memory.

    Returns a list of dicts: {"doc_id", "text", "path"}
    """
    patterns = ["**/*.pdf", "**/*.txt"]
    paths: List[str] = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(data_dir, pat), recursive=True))

    documents: List[Dict] = []
    for path in sorted(set(paths)):
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".pdf":
                text = _read_pdf(path)
            elif ext == ".txt":
                text = _read_txt(path)
            else:
                continue
        except Exception:
            # Skip unreadable files
            continue

        if not text or not text.strip():
            continue

        documents.append({"doc_id": os.path.basename(path), "text": text, "path": path})

    return documents


