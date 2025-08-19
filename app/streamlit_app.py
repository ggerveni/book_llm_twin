import os
import sys
from typing import List, Dict

# Ensure project root is on PYTHONPATH so `rag` can be imported when running from `app/`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
	sys.path.insert(0, ROOT_DIR)

import streamlit as st
import logging
from dotenv import load_dotenv

from rag.retriever import QdrantRetriever
from rag.generator import generate_answer, generate_answer_stream


load_dotenv()


st.set_page_config(page_title="Book LLM Twin", page_icon="📚", layout="wide")
st.title("📚 Book LLM Twin")
st.caption("Local RAG using Qdrant, Sentence-Transformers, and Ollama")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

with st.sidebar:
	st.header("Settings")

	collection_name = st.text_input("Qdrant collection", value=os.getenv("QDRANT_COLLECTION", ""))

	# Embedding model is fixed via .env; no UI selection
	embedding_model = os.getenv("EMBEDDING_MODEL", "")
	if not embedding_model:
		st.warning("Set EMBEDDING_MODEL in your .env file")
	else:
		st.caption(f"Embedding model (from .env): {embedding_model}")

	# Ollama model is fixed via .env; no UI selection
	ollama_model = os.getenv("OLLAMA_MODEL", "")
	if not ollama_model:
		st.warning("Set OLLAMA_MODEL in your .env file")
	else:
		st.caption(f"Ollama model (from .env): {ollama_model}")

	top_k = st.slider("Top-K (retrieval)", 1, 10, int(os.getenv("TOP_K", "5")))
	score_threshold = st.slider("Score threshold (optional)", 0.0, 1.0, 0.0, 0.01)

	st.divider()
	st.subheader("Generation")
	num_predict = st.slider("Max tokens (num_predict)", 32, 1024, int(os.getenv("OLLAMA_NUM_PREDICT", "256")), 32)
	temperature = st.slider("Temperature", 0.0, 1.5, float(os.getenv("OLLAMA_TEMPERATURE", "0.2")), 0.05)
	stream_mode = st.toggle("Stream tokens", value=True)

	st.subheader("Prompt limits")
	prompt_max_contexts = st.number_input("Max contexts", 1, 10, int(os.getenv("PROMPT_MAX_CONTEXTS", "4")))
	prompt_max_chars = st.number_input("Max chars per context", 200, 5000, int(os.getenv("PROMPT_MAX_CHARS_PER_CONTEXT", "1200")), 50)

query = st.text_area("Enter your question (English):", height=120)
ask = st.button("Ask")


def _to_ctx(retrieved) -> List[Dict[str, str]]:
	return [{"text": r.text, "source": r.source} for r in retrieved]


if ask and query.strip():
	status = st.status("Contacting Qdrant and searching...", state="running")
	try:
		retriever = QdrantRetriever(collection_name=collection_name, embedding_model_name=embedding_model, top_k=top_k)
		hits = retriever.retrieve(query=query, score_threshold=(score_threshold or None))
		status.update(label="Qdrant search completed", state="complete")
	except Exception as e:
		status.update(label=f"Qdrant search failed: {e}", state="error")
		raise

	st.subheader("Answer")
	ctx = _to_ctx(hits)
	os.environ["PROMPT_MAX_CONTEXTS"] = str(int(prompt_max_contexts))
	os.environ["PROMPT_MAX_CHARS_PER_CONTEXT"] = str(int(prompt_max_chars))
	os.environ["OLLAMA_NUM_PREDICT"] = str(int(num_predict))
	os.environ["OLLAMA_TEMPERATURE"] = str(float(temperature))

	answer_container = st.empty()
	if stream_mode:
		streamed_text = []
		gen_status = st.status("Generating answer...", state="running")
		try:
			for chunk in generate_answer_stream(
				question=query,
				contexts=ctx,
				model=ollama_model,
				max_tokens=int(num_predict),
				temperature=float(temperature),
			):
				streamed_text.append(chunk)
				answer_container.write("".join(streamed_text))
			gen_status.update(label="Answer generation completed", state="complete")
		except Exception as e:
			gen_status.update(label=f"Answer generation failed: {e}", state="error")
	else:
		gen_status = st.status("Generating answer...", state="running")
		try:
			answer = generate_answer(
				question=query,
				contexts=ctx,
				model=ollama_model,
				max_tokens=int(num_predict),
			)
			answer_container.write(answer)
			gen_status.update(label="Answer generation completed", state="complete")
		except Exception as e:
			gen_status.update(label=f"Answer generation failed: {e}", state="error")

	st.subheader("Sources")
	for i, h in enumerate(hits, start=1):
		with st.expander(f"[{i}] {h.source} (score: {h.score:.3f})"):
			st.write(h.text)




