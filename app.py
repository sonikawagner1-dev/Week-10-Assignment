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
MEMORY_PATH = Path("memory.json")


def load_hf_token() -> str | None:
    try:
        token = st.secrets["HF_TOKEN"]
    except Exception:
        return None

    token = token.strip()
    return token or None


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
    chat_path(chat["id"]).write_text(json.dumps(chat, indent=2), encoding="utf-8")


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


def load_memory() -> dict:
    if not MEMORY_PATH.exists():
        return {}

    try:
        data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    return data if isinstance(data, dict) else {}


def save_memory(memory: dict) -> None:
    MEMORY_PATH.write_text(json.dumps(memory, indent=2), encoding="utf-8")


def clear_memory() -> None:
    st.session_state.memory = {}
    save_memory({})


def merge_memory(existing: dict, incoming: dict) -> dict:
    merged = dict(existing)

    for key, value in incoming.items():
        if value in (None, "", [], {}):
            continue

        if isinstance(value, list):
            current = merged.get(key, [])
            if not isinstance(current, list):
                current = [current] if current not in (None, "", {}) else []
            for item in value:
                if item not in current:
                    current.append(item)
            merged[key] = current
        elif isinstance(value, dict):
            current = merged.get(key, {})
            if isinstance(current, dict):
                current.update(value)
                merged[key] = current
            else:
                merged[key] = value
        else:
            merged[key] = value

    return merged


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def filter_memory_to_message(user_message: str, extracted: dict) -> dict:
    message_text = normalize_text(user_message)
    filtered = {}

    for key, value in extracted.items():
        if isinstance(value, str):
            if normalize_text(value) in message_text:
                filtered[key] = value
        elif isinstance(value, list):
            kept_items = [
                item for item in value
                if isinstance(item, str) and normalize_text(item) in message_text
            ]
            if kept_items:
                filtered[key] = kept_items
        elif isinstance(value, dict):
            nested = filter_memory_to_message(user_message, value)
            if nested:
                filtered[key] = nested

    return filtered


def build_system_prompt(memory: dict) -> str:
    if not memory:
        return "You are a helpful, concise assistant."

    memory_blob = json.dumps(memory, ensure_ascii=True)
    return (
        "You are a helpful, concise assistant. Use the stored user memory when it is "
        "relevant to personalize responses, but do not mention the memory store directly. "
        f"Stored user memory: {memory_blob}"
    )


def build_model_messages(messages: list[dict[str, str]], memory: dict) -> list[dict[str, str]]:
    return [{"role": "system", "content": build_system_prompt(memory)}, *messages]


def parse_json_object(text: str) -> dict:
    stripped = text.strip()

    if stripped.startswith("```"):
        fence_parts = stripped.split("```")
        for part in fence_parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                stripped = candidate
                break

    start = stripped.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", stripped, 0)

    depth = 0
    end = None
    for index, char in enumerate(stripped[start:], start=start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break

    if end is None:
        raise json.JSONDecodeError("Incomplete JSON object", stripped, start)

    extracted = json.loads(stripped[start:end])
    return extracted if isinstance(extracted, dict) else {}


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
            if not raw_line or not raw_line.startswith("data:"):
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


def extract_memory_from_message(user_message: str, hf_token: str) -> dict:
    headers = {"Authorization": f"Bearer {hf_token}"}
    extraction_prompt = (
        "Given only this user message, extract personal facts or preferences that are explicitly "
        "stated in the message. Do not infer, guess, generalize, or add unstated traits. "
        "Use compact JSON with useful keys like name, interests, favorite_topics, "
        "preferred_language, communication_style, or preferences when directly supported. "
        "If there are no explicit personal facts, return {}. Return JSON only.\n\n"
        f"User message: {user_message}"
    )
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": extraction_prompt}],
        "max_tokens": 200,
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"].strip()
    extracted = parse_json_object(content)
    return filter_memory_to_message(user_message, extracted)


def ensure_app_state() -> None:
    if "chats" not in st.session_state or "active_chat_id" not in st.session_state:
        chats = load_saved_chats()
        st.session_state.chats = chats
        st.session_state.active_chat_id = chats[0]["id"] if chats else None

    if "memory" not in st.session_state:
        st.session_state.memory = load_memory()


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
    elif st.session_state.active_chat_id == chat_id:
        st.session_state.active_chat_id = remaining_chats[0]["id"]


st.set_page_config(page_title="My AI Chat", layout="wide")

ensure_app_state()

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

    chat_list = st.container(height=360)
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

    with st.expander("User Memory", expanded=True):
        if st.session_state.memory:
            st.json(st.session_state.memory)
        else:
            st.caption("No saved memory yet.")

        if st.button("Clear Memory", use_container_width=True):
            clear_memory()
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

    request_messages = build_model_messages(active_chat["messages"], st.session_state.memory)

    with chat_container:
        with st.chat_message("assistant"):
            try:
                reply = st.write_stream(stream_model_response(request_messages, hf_token))
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

    try:
        extracted_memory = extract_memory_from_message(prompt, hf_token)
        if extracted_memory:
            st.session_state.memory = merge_memory(st.session_state.memory, extracted_memory)
            save_memory(st.session_state.memory)
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        pass

    st.rerun()
