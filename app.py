import json
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import requests
import streamlit as st


API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
CHATS_DIR = Path("chats")


def load_hf_token() -> str | None:
    try:
        token = st.secrets["HF_TOKEN"]
    except Exception:
        return None

    token = token.strip()
    return token or None


def stream_model_response(messages: list[dict[str, str]], hf_token: str):
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 512,
        "stream": True,
    }

    with requests.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=30,
        stream=True,
    ) as response:
        response.raise_for_status()

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if not raw_line.startswith("data:"):
                continue

            data_line = raw_line.removeprefix("data:").strip()
            if data_line == "[DONE]":
                break

            chunk = json.loads(data_line)
            choices = chunk.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content
                time.sleep(0.02)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def format_timestamp(timestamp: str) -> str:
    try:
        return datetime.fromisoformat(timestamp).strftime("%b %d, %I:%M %p")
    except ValueError:
        return timestamp


def build_chat_title(messages: list[dict[str, str]]) -> str:
    for message in messages:
        if message["role"] == "user":
            content = message["content"].strip()
            if content:
                return content[:30] + ("..." if len(content) > 30 else "")
    return "New Chat"


def chat_path(chat_id: str) -> Path:
    return CHATS_DIR / f"{chat_id}.json"


def save_chat(chat: dict[str, str | list[dict[str, str]]]) -> None:
    CHATS_DIR.mkdir(exist_ok=True)
    path = chat_path(chat["id"])
    path.write_text(json.dumps(chat, indent=2), encoding="utf-8")


def load_saved_chats() -> list[dict[str, str | list[dict[str, str]]]]:
    CHATS_DIR.mkdir(exist_ok=True)
    chats = []

    for path in sorted(CHATS_DIR.glob("*.json")):
        try:
            chat = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if not all(key in chat for key in ("id", "title", "timestamp", "messages")):
            continue

        if not isinstance(chat["messages"], list):
            continue

        chats.append(chat)

    chats.sort(key=lambda chat: chat["timestamp"], reverse=True)
    return chats


def create_chat() -> dict[str, str | list[dict[str, str]]]:
    timestamp = now_iso()
    return {
        "id": str(uuid4()),
        "title": "New Chat",
        "timestamp": timestamp,
        "messages": [],
    }


def ensure_chat_state() -> None:
    if "chats" in st.session_state and "active_chat_id" in st.session_state:
        return

    chats = load_saved_chats()
    st.session_state.chats = chats
    st.session_state.active_chat_id = chats[0]["id"] if chats else None


def get_active_chat() -> dict[str, str | list[dict[str, str]]] | None:
    active_chat_id = st.session_state.get("active_chat_id")
    for chat in st.session_state.chats:
        if chat["id"] == active_chat_id:
            return chat
    return None


def set_active_chat(chat_id: str) -> None:
    st.session_state.active_chat_id = chat_id


def add_new_chat() -> None:
    new_chat = create_chat()
    st.session_state.chats.insert(0, new_chat)
    st.session_state.active_chat_id = new_chat["id"]
    save_chat(new_chat)


def delete_chat(chat_id: str) -> None:
    path = chat_path(chat_id)
    if path.exists():
        path.unlink()

    remaining_chats = [chat for chat in st.session_state.chats if chat["id"] != chat_id]
    st.session_state.chats = remaining_chats

    if not remaining_chats:
        st.session_state.active_chat_id = None
        return

    if st.session_state.active_chat_id == chat_id:
        st.session_state.active_chat_id = remaining_chats[0]["id"]


st.set_page_config(page_title="My AI Chat", layout="wide")

ensure_chat_state()

st.title("My AI Chat")
st.write("Chat with the model using the Hugging Face Inference Router.")

hf_token = load_hf_token()
if not hf_token:
    st.error(
        "Missing Hugging Face token. Add `HF_TOKEN` to `.streamlit/secrets.toml` "
        "or to the Streamlit Cloud Secrets settings."
    )
    st.stop()

with st.sidebar:
    st.header("Chats")

    if st.button("New Chat", use_container_width=True):
        add_new_chat()
        st.rerun()

    chat_list = st.container(height=500)
    with chat_list:
        for chat in st.session_state.chats:
            is_active = chat["id"] == st.session_state.active_chat_id
            columns = st.columns([5, 1], vertical_alignment="center")

            chat_label = f"{chat['title']}\n{format_timestamp(chat['timestamp'])}"
            if columns[0].button(
                chat_label,
                key=f"select_{chat['id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                set_active_chat(chat["id"])
                st.rerun()

            if columns[1].button("✕", key=f"delete_{chat['id']}", use_container_width=True):
                delete_chat(chat["id"])
                st.rerun()

active_chat = get_active_chat()

if active_chat is None:
    st.info("No chats yet. Create a new chat from the sidebar.")
    st.stop()

chat_container = st.container(height=500)
with chat_container:
    for message in active_chat["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

if not active_chat["messages"]:
    st.info("Start a new conversation using the message box below.")

prompt = st.chat_input("Type your message")

if prompt:
    user_message = {"role": "user", "content": prompt}
    active_chat["messages"].append(user_message)
    active_chat["title"] = build_chat_title(active_chat["messages"])
    active_chat["timestamp"] = now_iso()
    save_chat(active_chat)

    with chat_container:
        with st.chat_message("user"):
            st.write(prompt)

    with chat_container:
        with st.chat_message("assistant"):
            try:
                reply = st.write_stream(stream_model_response(active_chat["messages"], hf_token))
                if not isinstance(reply, str) or not reply.strip():
                    raise ValueError("Empty streamed response")
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else "unknown"
                if status_code == 401:
                    reply = "Invalid Hugging Face token. Check the `HF_TOKEN` value and try again."
                elif status_code == 429:
                    reply = "Rate limit reached. Wait a moment and try again."
                else:
                    reply = f"API request failed with status code {status_code}."
                st.error(reply)
            except requests.RequestException:
                reply = "Network error: the app could not reach the Hugging Face API."
                st.error(reply)
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
                reply = "The API returned an unexpected response format."
                st.error(reply)

    active_chat["messages"].append({"role": "assistant", "content": reply})
    active_chat["timestamp"] = now_iso()
    save_chat(active_chat)
    st.rerun()
