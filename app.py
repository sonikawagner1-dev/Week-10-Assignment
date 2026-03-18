import requests
import streamlit as st


API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
TEST_MESSAGE = "Hello!"


def load_hf_token() -> str | None:
    try:
        token = st.secrets["HF_TOKEN"]
    except Exception:
        return None

    token = token.strip()
    return token or None


def get_model_response(hf_token: str) -> str:
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": TEST_MESSAGE}],
        "max_tokens": 512,
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


st.set_page_config(page_title="My AI Chat", layout="wide")

st.title("My AI Chat")
st.write("This app sends a single test message to the Hugging Face Inference Router.")
st.write(f"Test prompt: `{TEST_MESSAGE}`")

hf_token = load_hf_token()
if not hf_token:
    st.error(
        "Missing Hugging Face token. Add `HF_TOKEN` to `.streamlit/secrets.toml` "
        "or to the Streamlit Cloud Secrets settings."
    )
    st.stop()

with st.spinner("Sending test message..."):
    try:
        model_reply = get_model_response(hf_token)
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        if status_code == 401:
            st.error("Invalid Hugging Face token. Check the `HF_TOKEN` value and try again.")
        elif status_code == 429:
            st.error("Rate limit reached. Wait a moment and try again.")
        else:
            st.error(f"API request failed with status code {status_code}.")
        st.stop()
    except requests.RequestException:
        st.error("Network error: the app could not reach the Hugging Face API.")
        st.stop()
    except (KeyError, IndexError, TypeError, ValueError):
        st.error("The API returned an unexpected response format.")
        st.stop()

st.subheader("Model Response")
st.write(model_reply)
