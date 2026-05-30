import os
import json
import time
from typing import List, Dict, Any, Tuple, TypedDict

import numpy as np
import faiss
import gradio as gr
import requests
from sentence_transformers import SentenceTransformer

from langgraph.graph import StateGraph, END

KB_DIR = r".\kb_store"
MODEL_DIR = r".\sentence-transformer\sentence-transformer\all-MiniLM-L6-v2"

MODEL_CONFIGS = {
    "Gemma": {
        "base_url": "http://MB-QS-PP-H200.int.pg.com:8842/v1",
        "model": "google/gemma-3-27b-it",
    },
    "Pixtral": {
        "base_url": "http://MB-QS-PP-H200.int.pg.com:8844/v1",
        "model": "mistralai/Pixtral-12B-2409",
    },
    "Qwen3-VL": {
        "base_url": "http://MB-QS-PP-H200.int.pg.com:8841/v1",
        "model": "Qwen/Qwen3-VL-8B-Instruct",
    },
    "InternVL3.5": {
        "base_url": "http://MB-QS-PP-H200.int.pg.com:8843/v1",
        "model": "OpenGVLab/InternVL3_5-14B",
    },
}

LLM_API_KEY = os.getenv("LLM_API_KEY", "")

MAX_CONTEXT_CHARS = 12000
TIMEOUT_SEC = 180
RETRIES = 2


def paths():
    return {
        "index": os.path.join(KB_DIR, "faiss.index"),
        "meta": os.path.join(KB_DIR, "meta.jsonl"),
    }


def load_meta() -> List[Dict[str, Any]]:
    rows = []
    with open(paths()["meta"], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_index() -> faiss.Index:
    return faiss.read_index(paths()["index"])


def validate_startup():
    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(f"MODEL_DIR not found: {MODEL_DIR}")

    if not os.path.exists(paths()["index"]) or not os.path.exists(paths()["meta"]):
        raise RuntimeError("KB not found. Run: python ingest_kb_3.py")


validate_startup()

print("[INFO] Loading embedding model, FAISS index, and metadata once...")
GLOBAL_EMBEDDER = SentenceTransformer(MODEL_DIR)
GLOBAL_INDEX = load_index()
GLOBAL_META_ROWS = load_meta()
print("[INFO] App resources loaded.")

def embed_query(embedder: SentenceTransformer, q: str) -> np.ndarray:
    v = embedder.encode(
        [q],
        convert_to_numpy=True,
        show_progress_bar=False
    ).astype("float32")
    faiss.normalize_L2(v)
    return v


def retrieve(index, meta_rows, embedder, query: str, k: int):
    qv = embed_query(embedder, query)
    scores, idxs = index.search(qv, int(k))

    hits = []
    for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
        if 0 <= idx < len(meta_rows):
            row = dict(meta_rows[idx])
            row["score"] = float(score)
            hits.append(row)

    return hits


def deduplicate_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []

    for h in hits:
        key = (h.get("path", ""), h.get("chunk_id", -1))
        if key not in seen:
            seen.add(key)
            unique.append(h)

    unique.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return unique


def build_context(hits: List[Dict[str, Any]]) -> str:
    blocks = []
    total = 0

    for h in hits:
        src = h.get("relative_path", h.get("source", "unknown"))
        cid = h.get("chunk_id", -1)
        txt = (h.get("text") or "").strip()

        block = f"[SOURCE: {src} | chunk {cid}]\n{txt}\n"

        if total + len(block) > MAX_CONTEXT_CHARS:
            break

        blocks.append(block)
        total += len(block)

    return "\n---\n".join(blocks)


def build_source_choices(hits: List[Dict[str, Any]]):
    choices = []

    for h in hits:
        full_path = h.get("path", "")
        rel = h.get("relative_path", h.get("source", "unknown"))
        score = h.get("score", 0.0)
        chunk = h.get("chunk_id", -1)

        label = f"{rel} (chunk {chunk}, score={score:.3f})"
        choices.append((label, full_path))

    return choices


def open_selected_folder(selected_path: str):
    if not selected_path:
        return

    try:
        folder_path = os.path.dirname(selected_path)
        os.startfile(folder_path)
    except Exception:
        pass


def openai_style_call(
    llm_choice: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> Tuple[str, str]:
    config = MODEL_CONFIGS[llm_choice]
    url = config["base_url"].rstrip("/") + "/chat/completions"

    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY.strip():
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    payload = {
        "model": config["model"],
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SEC)

    if r.status_code != 200:
        return f"LLM error ({llm_choice}) {r.status_code}: {r.text}", "error"

    data = r.json()
    text = data["choices"][0]["message"]["content"]
    finish_reason = data["choices"][0].get("finish_reason", "")

    return text, finish_reason

def history_for_rewrite(chat_history):
    pairs = []
    current_user = None

    for msg in chat_history or []:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "user":
            current_user = content
        elif role == "assistant" and current_user is not None:
            pairs.append((current_user, content))
            current_user = None

    return pairs


def format_recent_history(pair_history, history_window: int) -> str:
    if not pair_history or int(history_window) <= 0:
        return ""

    recent = pair_history[-int(history_window):]
    lines = []

    for user_msg, assistant_msg in recent:
        if user_msg:
            lines.append(f"User: {user_msg}")
        if assistant_msg:
            lines.append(f"Assistant: {assistant_msg[:800]}")

    return "\n".join(lines)


class AgentState(TypedDict):
    original_question: str
    rewritten_question: str
    retrieval_queries: List[str]
    hits: List[Dict[str, Any]]
    context: str
    answer: str
    source_choices: List[Tuple[str, str]]
    agent_trace: str

    pair_history: list
    llm_choice: str
    top_k: int
    temperature: float
    max_tokens: int
    use_history_rewrite: bool
    history_window: int
    agent_min_score: float
    agent_min_hits: int


def node_rewrite_question(state: AgentState) -> AgentState:
    question = state["original_question"]
    trace = ["LangGraph Agentic Mode: ON"]

    if not state["use_history_rewrite"] or state["history_window"] <= 0 or not state["pair_history"]:
        state["rewritten_question"] = question
        trace.append("Question rewrite: skipped; using original question.")
        state["agent_trace"] = "\n".join(trace)
        return state

    recent_history = format_recent_history(state["pair_history"], state["history_window"])

    system = (
        "You rewrite follow-up questions into standalone search queries for a RAG system. "
        "Do not answer the question. Only rewrite it. "
        "If the current question is already standalone, return it unchanged. "
        "Keep it concise and specific."
    )

    user = (
        f"RECENT CHAT HISTORY:\n{recent_history}\n\n"
        f"CURRENT QUESTION:\n{question}\n\n"
        "Standalone rewritten question:"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    try:
        rewritten, finish_reason = openai_style_call(
            llm_choice=state["llm_choice"],
            messages=messages,
            temperature=0.0,
            max_tokens=256,
        )

        if finish_reason == "error" or not rewritten.strip():
            rewritten = question

        state["rewritten_question"] = rewritten.strip()
        trace.append(f"Question rewrite: {state['rewritten_question']}")

    except Exception:
        state["rewritten_question"] = question
        trace.append("Question rewrite failed; using original question.")

    state["agent_trace"] = "\n".join(trace)
    return state


def node_plan_queries(state: AgentState) -> AgentState:
    question = state["rewritten_question"]
    trace = state["agent_trace"].splitlines()

    recent_history = format_recent_history(state["pair_history"], state["history_window"])

    system = (
        "You are a query planning agent for a document RAG system. "
        "Create 2 or 3 short search queries that retrieve relevant document chunks. "
        "Do not answer the question. "
        "Return only the search queries, one per line."
    )

    user = (
        f"RECENT CHAT HISTORY:\n{recent_history}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        "Create 2 or 3 retrieval search queries:"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    try:
        text, finish_reason = openai_style_call(
            llm_choice=state["llm_choice"],
            messages=messages,
            temperature=0.0,
            max_tokens=256,
        )

        if finish_reason == "error" or not text.strip():
            queries = [question]
        else:
            queries = []
            for line in text.splitlines():
                line = line.strip()
                line = line.lstrip("-").strip()
                line = line.lstrip("0123456789. )").strip()
                if line and line not in queries:
                    queries.append(line)

            if not queries:
                queries = [question]

            queries = queries[:3]

    except Exception:
        queries = [question]

    state["retrieval_queries"] = queries

    trace.append("Planned retrieval queries:")
    for q in queries:
        trace.append(f"- {q}")

    state["agent_trace"] = "\n".join(trace)
    return state


def node_retrieve(state: AgentState) -> AgentState:
    trace = state["agent_trace"].splitlines()

    all_hits = []

    for q in state["retrieval_queries"]:
        hits = retrieve(
            index=GLOBAL_INDEX,
            meta_rows=GLOBAL_META_ROWS,
            embedder=GLOBAL_EMBEDDER,
            query=q,
            k=state["top_k"],
        )
        all_hits.extend(hits)

    unique_hits = deduplicate_hits(all_hits)
    unique_hits = unique_hits[:max(state["top_k"], state["agent_min_hits"])]

    state["hits"] = unique_hits

    top_score = unique_hits[0]["score"] if unique_hits else 0.0
    trace.append(f"Initial retrieval: {len(unique_hits)} unique chunks, top score={top_score:.3f}")

    state["agent_trace"] = "\n".join(trace)
    return state


def should_retry_retrieval(state: AgentState) -> str:
    hits = state["hits"]

    if not hits:
        return "retry"

    top_score = hits[0].get("score", 0.0)

    if top_score < state["agent_min_score"]:
        return "retry"

    if len(hits) < state["agent_min_hits"]:
        return "retry"

    return "continue"


def node_retry_retrieval(state: AgentState) -> AgentState:
    trace = state["agent_trace"].splitlines()

    retry_k = min(20, max(state["top_k"] + 3, state["agent_min_hits"] + 3))

    trace.append(
        f"Retrieval judge: weak retrieval detected. Retrying with rewritten question and top_k={retry_k}."
    )

    retry_hits = retrieve(
        index=GLOBAL_INDEX,
        meta_rows=GLOBAL_META_ROWS,
        embedder=GLOBAL_EMBEDDER,
        query=state["rewritten_question"],
        k=retry_k,
    )

    unique_hits = deduplicate_hits(state["hits"] + retry_hits)
    unique_hits = unique_hits[:retry_k]

    state["hits"] = unique_hits

    top_score = unique_hits[0]["score"] if unique_hits else 0.0
    trace.append(f"After retry: {len(unique_hits)} unique chunks, top score={top_score:.3f}")

    state["agent_trace"] = "\n".join(trace)
    return state


def node_build_context(state: AgentState) -> AgentState:
    context = build_context(state["hits"])
    source_choices = build_source_choices(state["hits"])

    state["context"] = context
    state["source_choices"] = source_choices

    trace = state["agent_trace"].splitlines()
    trace.append(f"Context built from {len(state['hits'])} chunks.")
    state["agent_trace"] = "\n".join(trace)

    return state


def node_answer(state: AgentState) -> AgentState:
    if not state["context"].strip():
        state["answer"] = "No relevant chunks retrieved. Try increasing top-k."
        return state

    system = (
        "You are a helpful technical assistant for question answering over retrieved documents. "
        "Answer directly, completely, and in a well-organized way. "
        "Use the retrieved context first. If the retrieved context is incomplete, briefly state that and then continue with general knowledge when useful. "
        "Do not hallucinate exact document-specific facts that are not supported by context. "
        "For technical/scientific questions, provide practical explanations, clear steps, and relevant terminology. "
        "If equations, mathematical expressions, formulas, metrics, or quantitative relationships are useful, include them. "
        "If the user asks for a diagram, flowchart, pipeline, workflow, architecture, or step-by-step process, provide a compact ASCII/text-based diagram or flowchart. "
        "If comparing methods, use a clear table when helpful. "
        "Avoid unnecessary introductions. "
        "Do not stop early. Make sure all major parts of the user's request are addressed."
    )

    user = (
        f"CONTEXT:\n{state['context']}\n\n"
        f"QUESTION:\n{state['rewritten_question']}\n\n"
        "Instructions:\n"
        "- Give a complete and useful answer.\n"
        "- Use retrieved context first.\n"
        "- If context is incomplete, say so briefly and then continue with general knowledge if helpful.\n"
        "- Include equations/formulas when they improve the answer.\n"
        "- Include compact text-based diagrams or flowcharts when the question asks for workflow, architecture, process, or pipeline.\n"
        "- Use tables for comparisons when helpful.\n"
        "- Keep the answer organized with short headings or bullets.\n"
        "- Do not end mid-answer."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    last_err = None

    for attempt in range(RETRIES + 1):
        try:
            full_answer_parts = []
            max_continuations = 5

            for _ in range(max_continuations):
                text, finish_reason = openai_style_call(
                    llm_choice=state["llm_choice"],
                    messages=messages,
                    temperature=state["temperature"],
                    max_tokens=state["max_tokens"],
                )

                if finish_reason == "error":
                    state["answer"] = text
                    return state

                full_answer_parts.append(text)

                if finish_reason != "length":
                    final_answer = "".join(full_answer_parts)
                    state["answer"] = (
                        f"{final_answer}\n\n"
                        f"---\n"
                        f"*Retrieval question used:* {state['rewritten_question']}"
                    )
                    return state

                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": (
                        "Continue exactly from where you stopped. "
                        "Do not repeat earlier text. "
                        "Finish the answer completely. "
                        "If a formula, diagram, table, or flowchart was requested but not yet completed, include it now."
                    )
                })

        except requests.exceptions.ReadTimeout:
            last_err = f"ReadTimeout after {TIMEOUT_SEC}s (attempt {attempt + 1}/{RETRIES + 1})"
        except Exception as e:
            last_err = f"Error calling {state['llm_choice']}: {e}"

        time.sleep(1.5 * (attempt + 1))

    state["answer"] = last_err or f"Unknown error calling {state['llm_choice']}."
    return state

workflow = StateGraph(AgentState)

workflow.add_node("rewrite_question", node_rewrite_question)
workflow.add_node("plan_queries", node_plan_queries)
workflow.add_node("retrieve", node_retrieve)
workflow.add_node("retry_retrieval", node_retry_retrieval)
workflow.add_node("build_context", node_build_context)
workflow.add_node("answer", node_answer)

workflow.set_entry_point("rewrite_question")

workflow.add_edge("rewrite_question", "plan_queries")
workflow.add_edge("plan_queries", "retrieve")

workflow.add_conditional_edges(
    "retrieve",
    should_retry_retrieval,
    {
        "retry": "retry_retrieval",
        "continue": "build_context",
    },
)

workflow.add_edge("retry_retrieval", "build_context")
workflow.add_edge("build_context", "answer")
workflow.add_edge("answer", END)

agent_graph = workflow.compile()

def run_langgraph_agent(
    message,
    pair_history,
    llm_choice,
    top_k,
    temperature,
    max_tokens,
    use_history_rewrite,
    history_window,
    agent_min_score,
    agent_min_hits,
):
    initial_state: AgentState = {
        "original_question": message.strip(),
        "rewritten_question": message.strip(),
        "retrieval_queries": [],
        "hits": [],
        "context": "",
        "answer": "",
        "source_choices": [],
        "agent_trace": "",

        "pair_history": pair_history,
        "llm_choice": llm_choice,
        "top_k": int(top_k),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "use_history_rewrite": bool(use_history_rewrite),
        "history_window": int(history_window),
        "agent_min_score": float(agent_min_score),
        "agent_min_hits": int(agent_min_hits),
    }

    final_state = agent_graph.invoke(initial_state)

    return (
        final_state["answer"],
        final_state["source_choices"],
        final_state["agent_trace"],
    )


def chat_submit(
    message,
    chat_history,
    llm_choice,
    top_k,
    temperature,
    max_tokens,
    use_history_rewrite,
    history_window,
    agent_min_score,
    agent_min_hits,
):
    if chat_history is None:
        chat_history = []

    if not message or not message.strip():
        return chat_history, "", gr.update(), ""

    pair_history = history_for_rewrite(chat_history)

    answer, source_choices, agent_trace = run_langgraph_agent(
        message=message,
        pair_history=pair_history,
        llm_choice=llm_choice,
        top_k=top_k,
        temperature=temperature,
        max_tokens=max_tokens,
        use_history_rewrite=use_history_rewrite,
        history_window=history_window,
        agent_min_score=agent_min_score,
        agent_min_hits=agent_min_hits,
    )

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": answer})

    return chat_history, "", gr.update(choices=source_choices, value=None), agent_trace

with gr.Blocks() as demo:
    gr.Markdown("# Light-weight Agentic Conversational RAG App")

    with gr.Row():
        with gr.Column(scale=1):
            model_choice = gr.Dropdown(
                choices=["Gemma", "Pixtral", "Qwen3-VL", "InternVL3.5"],
                value="Gemma",
                label="Model",
            )

            top_k = gr.Slider(
                1, 20,
                value=6,
                step=1,
                label="Top-k / K Neighbours",
                info="How many document chunks to retrieve from FAISS.",
            )

            history_window = gr.Slider(
                0, 10,
                value=2,
                step=1,
                label="Search Window / Chat History Window",
                info="How many previous conversation turns to use for rewriting follow-up questions. 0 = independent questions.",
            )

            temperature = gr.Slider(
                0.0, 1.0,
                value=0.2,
                step=0.05,
                label="Temperature",
                info="Lower = more factual, higher = more creative.",
            )

            max_tokens = gr.Slider(
                128, 2048,
                value=768,
                step=32,
                label="Max tokens",
                info="Maximum answer length.",
            )

            use_history_rewrite = gr.Checkbox(
                value=True,
                label="Use chat history for follow-up questions",
                info="Rewrites follow-up questions into standalone retrieval questions.",
            )

            agent_min_score = gr.Slider(
                0.0, 1.0,
                value=0.25,
                step=0.05,
                label="Agent Min Retrieval Score",
            )

            agent_min_hits = gr.Slider(
                1, 10,
                value=3,
                step=1,
                label="Agent Min Unique Chunks",
            )

            gr.Markdown(
                "Light-weight agentic nodes: rewrite → plan queries → retrieve → judge/retry → answer."
            )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=650,
            )

            chat_input = gr.Textbox(
                label="Question",
                placeholder="Ask a question about your ingested documents...",
                lines=2,
            )

            with gr.Row():
                submit_chat_btn = gr.Button("Submit", variant="primary")
                clear_chat_btn = gr.Button("Clear chat")

            source_selector = gr.Radio(
                label="Sources (click one to open the corresponding file location)",
                choices=[],
            )

            agent_trace_box = gr.Textbox(
                label="LangGraph Agent Trace",
                lines=12,
            )

    submit_chat_btn.click(
        fn=chat_submit,
        inputs=[
            chat_input,
            chatbot,
            model_choice,
            top_k,
            temperature,
            max_tokens,
            use_history_rewrite,
            history_window,
            agent_min_score,
            agent_min_hits,
        ],
        outputs=[chatbot, chat_input, source_selector, agent_trace_box],
        show_progress="minimal",
    )

    chat_input.submit(
        fn=chat_submit,
        inputs=[
            chat_input,
            chatbot,
            model_choice,
            top_k,
            temperature,
            max_tokens,
            use_history_rewrite,
            history_window,
            agent_min_score,
            agent_min_hits,
        ],
        outputs=[chatbot, chat_input, source_selector, agent_trace_box],
        show_progress="minimal",
    )

    source_selector.change(
        fn=open_selected_folder,
        inputs=[source_selector],
        outputs=[],
        show_progress="hidden",
    )

    clear_chat_btn.click(
        fn=lambda: ([], gr.update(choices=[], value=None), ""),
        inputs=[],
        outputs=[chatbot, source_selector, agent_trace_box],
        show_progress="hidden",
    )


demo.launch(server_name="127.0.0.2", share=False)
