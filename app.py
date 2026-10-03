import html
import importlib
import json
import os
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from urllib.request import Request, urlopen

import pytesseract
import streamlit as st
from pypdf import PdfReader
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

APP_DIR = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent
)
MODELS_DIR = APP_DIR / "models"
LOCAL_MODEL_FILENAME = "Llama-3.2-3B-Instruct-Q4_K_S.gguf"
LOCAL_MODEL_PATH = MODELS_DIR / LOCAL_MODEL_FILENAME
MODEL_DOWNLOAD_URL = (
    "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/"
    "5ab33fa94d1d04e903623ae72c95d1696f09f9e8/"
    f"{LOCAL_MODEL_FILENAME}?download=true"
)
MODEL_PATHS = sorted(APP_DIR.glob("*.gguf")) + sorted(MODELS_DIR.glob("*.gguf"))
if LOCAL_MODEL_PATH in MODEL_PATHS:
    MODEL_PATHS.remove(LOCAL_MODEL_PATH)
    MODEL_PATHS.insert(0, LOCAL_MODEL_PATH)
LOCAL_ENGINE = "💻 Local Offline (Llama-3.2)"
GEMINI_ENGINE = "☁️ Gemini Cloud API (Instant)"

try:
    llama_cpp = importlib.import_module("llama_cpp")
    Llama = getattr(llama_cpp, "Llama", None)
except ImportError:
    Llama = None

MOCK_FLASHCARDS = [
    {
        "question": "What is the largest planet in our solar system?",
        "answer": "Jupiter."
    },
    {
        "question": "What process do plants use to convert sunlight into energy?",
        "answer": "Photosynthesis."
    },
    {
        "question": "What is the chemical symbol for gold?",
        "answer": "Au."
    },
    {
        "question": "How many sides does a hexagon have?",
        "answer": "Six."
    },
    {
        "question": "Which ocean is the largest on Earth?",
        "answer": "The Pacific Ocean."
    }
]


def initialize_decks() -> None:
    if "decks" not in st.session_state:
        legacy_cards = st.session_state.get("flashcards", MOCK_FLASHCARDS)
        cards = legacy_cards if isinstance(legacy_cards, list) and legacy_cards else MOCK_FLASHCARDS
        st.session_state.decks = {"Quick Start": [card.copy() for card in cards]}
        st.session_state.active_deck = "Quick Start"
        st.session_state.deck_positions = {
            "Quick Start": st.session_state.get("current_card", 0)
        }
        st.session_state.deck_show_answer = {
            "Quick Start": st.session_state.get("show_answer", False)
        }

    if not st.session_state.decks:
        st.session_state.decks = {"Quick Start": [card.copy() for card in MOCK_FLASHCARDS]}
    if st.session_state.get("active_deck") not in st.session_state.decks:
        st.session_state.active_deck = next(iter(st.session_state.decks))

    st.session_state.setdefault("deck_positions", {})
    st.session_state.setdefault("deck_show_answer", {})
    st.session_state.setdefault("custom_cards_added", 0)
    st.session_state.setdefault("manual_notice", "")
    st.session_state.setdefault("deck_notice", "")
    for deck_name, cards in st.session_state.decks.items():
        st.session_state.deck_positions.setdefault(deck_name, 0)
        st.session_state.deck_show_answer.setdefault(deck_name, False)
        st.session_state.deck_positions[deck_name] = min(
            max(st.session_state.deck_positions[deck_name], 0),
            max(len(cards) - 1, 0)
        )


def load_mock_deck() -> None:
    deck_name = st.session_state.active_deck
    st.session_state.decks[deck_name] = [card.copy() for card in MOCK_FLASHCARDS]
    st.session_state.deck_positions[deck_name] = 0
    st.session_state.deck_show_answer[deck_name] = False


def create_deck() -> None:
    deck_name = st.session_state.get("new_deck_name", "").strip()
    if not deck_name:
        st.session_state.deck_notice = "Enter a name for the new deck."
        return
    if deck_name in st.session_state.decks:
        st.session_state.deck_notice = "A deck with that name already exists."
        return

    st.session_state.decks[deck_name] = []
    st.session_state.deck_positions[deck_name] = 0
    st.session_state.deck_show_answer[deck_name] = False
    st.session_state.active_deck = deck_name
    st.session_state.deck_notice = f"Created {deck_name}."


def add_manual_card() -> None:
    question = st.session_state.get("manual_front", "").strip()
    answer = st.session_state.get("manual_back", "").strip()
    if not question or not answer:
        st.session_state.manual_notice = "Enter both a front and a back for this card."
        return

    deck_name = st.session_state.active_deck
    st.session_state.decks[deck_name].append({"question": question, "answer": answer})
    st.session_state.deck_positions[deck_name] = len(st.session_state.decks[deck_name]) - 1
    st.session_state.deck_show_answer[deck_name] = False
    st.session_state.custom_cards_added += 1
    st.session_state.manual_notice = f"Card added to {deck_name}."


def flip_card() -> None:
    deck_name = st.session_state.active_deck
    st.session_state.deck_show_answer[deck_name] = not st.session_state.deck_show_answer[deck_name]


def move_card(offset: int) -> None:
    deck_name = st.session_state.active_deck
    cards = st.session_state.decks[deck_name]
    st.session_state.deck_positions[deck_name] = min(
        max(st.session_state.deck_positions[deck_name] + offset, 0),
        max(len(cards) - 1, 0)
    )
    st.session_state.deck_show_answer[deck_name] = False


def clear_active_deck() -> None:
    deck_name = st.session_state.active_deck
    st.session_state.decks[deck_name] = []
    st.session_state.deck_positions[deck_name] = 0
    st.session_state.deck_show_answer[deck_name] = False
    st.session_state.deck_notice = f"Cleared {deck_name}."


def delete_current_card() -> None:
    deck_name = st.session_state.active_deck
    cards = st.session_state.decks[deck_name]
    if not cards:
        return

    current_position = min(
        max(st.session_state.deck_positions[deck_name], 0),
        len(cards) - 1
    )
    cards.pop(current_position)
    st.session_state.deck_positions[deck_name] = min(
        current_position,
        max(len(cards) - 1, 0)
    )
    st.session_state.deck_show_answer[deck_name] = False


def download_local_model(progress_bar: Any, status_text: Any) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    partial_path = LOCAL_MODEL_PATH.with_name(LOCAL_MODEL_PATH.name + ".part")
    request = Request(MODEL_DOWNLOAD_URL, headers={"User-Agent": "NomixeoStudyApp/1.0"})
    downloaded_bytes = 0
    chunk_size = 8 * 1024 * 1024

    try:
        with urlopen(request, timeout=60) as response, partial_path.open("wb") as model_file:
            total_bytes = int(response.headers.get("Content-Length") or 0)
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                model_file.write(chunk)
                downloaded_bytes += len(chunk)

                downloaded_gb = downloaded_bytes / (1024 ** 3)
                if total_bytes:
                    progress = min(downloaded_bytes / total_bytes, 1.0)
                    total_gb = total_bytes / (1024 ** 3)
                    progress_bar.progress(
                        progress,
                        text=f"Downloading model: {progress:.0%} ({downloaded_gb:.2f}/{total_gb:.2f} GB)"
                    )
                else:
                    progress_bar.progress(
                        0.0,
                        text=f"Downloaded {downloaded_gb:.2f} GB..."
                    )
                status_text.text(f"Downloaded {downloaded_gb:.2f} GB")

        if downloaded_bytes == 0:
            raise OSError("The model download returned an empty file.")
        if total_bytes and downloaded_bytes != total_bytes:
            raise OSError("The model download was incomplete. Please try again.")

        partial_path.replace(LOCAL_MODEL_PATH)
        progress_bar.progress(1.0, text="Model download complete.")
        status_text.text(f"Saved model to {LOCAL_MODEL_PATH}")
        return LOCAL_MODEL_PATH
    except Exception:
        if partial_path.exists():
            partial_path.unlink()
        raise


@st.cache_resource(show_spinner=False)
def load_local_model(model_path: str) -> Any:
    if Llama is None:
        raise RuntimeError(
            "llama-cpp-python is not installed. Run `pip install -r requirements.txt`."
        )
    n_threads = max(1, (os.cpu_count() or 1) // 2)
    return Llama(model_path=model_path, n_ctx=4096, n_threads=n_threads, verbose=False)


@st.cache_data(show_spinner=False)
def extract_image_text(image_bytes: bytes) -> str:
    with Image.open(BytesIO(image_bytes)) as image:
        extracted_text = pytesseract.image_to_string(image)
    return "\n".join(
        line.strip() for line in extracted_text.splitlines() if line.strip()
    )


def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def build_flashcard_prompt(
    source_text: str,
    card_count: int,
    difficulty: str
) -> str:
    study_material = source_text.strip() or "Use the attached image as the study material."
    return f"""Create exactly {card_count} {difficulty.lower()} flashcards from the study material.
Output ONLY one valid JSON array of objects. Every object must have string-valued "question" and "answer" keys.
Do not include conversational text, explanations, markdown formatting, or backticks before or after the JSON array.
Study material:
{study_material[:10000]}
"""


def parse_flashcard_response(response_text: str, card_count: int) -> list[dict[str, str]]:
    response_text = response_text or ""
    try:
        generated_cards = json.loads(response_text)
    except json.JSONDecodeError:
        generated_cards = None

    if not isinstance(generated_cards, list):
        array_match = re.search(r"\[.*\]", response_text, re.DOTALL)
        if array_match:
            try:
                generated_cards = json.loads(array_match.group(0))
            except json.JSONDecodeError:
                generated_cards = None

    fallback_card = {
        "question": "Review Extracted Concepts",
        "answer": response_text[:400].strip() or "The model did not return a usable response."
    }
    if not isinstance(generated_cards, list):
        return [fallback_card]

    cards = [
        {
            "question": str(item.get("question", "")).strip(),
            "answer": str(item.get("answer", "")).strip()
        }
        for item in generated_cards
        if isinstance(item, dict)
        and str(item.get("question", "")).strip()
        and str(item.get("answer", "")).strip()
    ]
    if not cards:
        return [fallback_card]
    return cards[:card_count]


def generate_local_flashcards(
    model_path: str,
    source_text: str,
    card_count: int,
    difficulty: str
) -> list[dict[str, str]]:
    model = load_local_model(model_path)
    prompt = build_flashcard_prompt(source_text, card_count, difficulty)
    response = model.create_completion(
        prompt=prompt,
        max_tokens=1400,
        temperature=0.3
    )
    response_text = str(response["choices"][0].get("text", ""))
    return parse_flashcard_response(response_text, card_count)


def generate_gemini_flashcards(
    api_key: str,
    source_text: str,
    card_count: int,
    difficulty: str,
    image_bytes: Optional[bytes] = None
) -> list[dict[str, str]]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise RuntimeError(
            "The Gemini SDK is not installed. Run `pip install -r requirements.txt`."
        ) from error

    client = genai.Client(api_key=api_key)
    prompt = build_flashcard_prompt(source_text, card_count, difficulty)
    config = types.GenerateContentConfig(response_mime_type="application/json")

    if image_bytes:
        with Image.open(BytesIO(image_bytes)) as uploaded_image:
            image = uploaded_image.convert("RGB")
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=[prompt, image],
                config=config
            )
    else:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=config
        )

    return parse_flashcard_response(response.text or "", card_count)

st.set_page_config(
    page_title="Nomixeo Study App",
    layout="wide"
)

st.markdown(
    """
    <style>
        :root {
            --nm-black: #05050A;
            --nm-surface: #0B0C15;
            --nm-cyan: #00D9FF;
            --nm-blue: #2875FF;
            --nm-violet: #7B3FFF;
            --nm-magenta: #E426E6;
            --nm-text: #F5F7FF;
            --nm-muted: #A7AEC4;
        }
        .stApp {
            color: var(--nm-text);
            background:
                linear-gradient(135deg, rgba(0, 217, 255, 0.035) 1px, transparent 1px),
                linear-gradient(45deg, rgba(226, 38, 230, 0.025) 1px, transparent 1px),
                linear-gradient(145deg, #05050A, #0D0D18 58%, #07070D);
            background-size: 72px 72px, 72px 72px, auto;
            background-attachment: fixed;
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        .block-container {
            max-width: 1120px;
            padding-top: 2.25rem;
            padding-bottom: 3rem;
        }
        body, button, input, textarea {
            font-family: 'Segoe UI', 'Trebuchet MS', sans-serif;
        }
        h1, h2, h3 {
            color: var(--nm-text);
            font-family: 'Segoe UI', 'Trebuchet MS', sans-serif;
        }
        .brand-lockup {
            margin: 0 0 2rem;
            padding: 0 0 1.5rem;
            border-bottom: 1px solid rgba(0, 217, 255, 0.2);
        }
        .brand-eyebrow {
            margin: 0 0 0.4rem;
            color: var(--nm-cyan);
            font: 700 0.8rem/1.3 'Segoe UI', sans-serif;
            text-transform: uppercase;
        }
        .brand-title {
            display: inline-block;
            margin: 0;
            color: var(--nm-text);
            background: linear-gradient(100deg, var(--nm-cyan), var(--nm-blue) 38%, var(--nm-violet) 70%, var(--nm-magenta));
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font: 700 2.65rem/1.15 'Segoe UI', 'Trebuchet MS', sans-serif;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(17, 19, 34, 0.96), rgba(7, 8, 15, 0.97));
            border-right: 1px solid rgba(0, 217, 255, 0.2);
            -webkit-backdrop-filter: blur(22px);
            backdrop-filter: blur(22px);
        }
        section[data-testid="stSidebar"] h2,
        .stMarkdown h2, .stMarkdown h3 {
            color: var(--nm-text);
        }
        label, .stMarkdown p, [data-testid="stWidgetLabel"] p {
            color: var(--nm-muted);
        }
        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            min-height: 2.8rem;
            border: 1px solid transparent;
            border-radius: 10px;
            color: var(--nm-text);
            background:
                linear-gradient(var(--nm-surface), var(--nm-surface)) padding-box,
                linear-gradient(105deg, var(--nm-cyan), var(--nm-blue), var(--nm-violet), var(--nm-magenta)) border-box;
            box-shadow: 0 0 16px rgba(40, 117, 255, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.06);
            font-weight: 600;
            transition: transform 180ms ease, box-shadow 180ms ease, background-color 180ms ease;
        }
        div.stButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            color: #FFFFFF;
            transform: translateY(-1px);
            box-shadow: 0 0 12px rgba(0, 217, 255, 0.35), 0 0 24px rgba(226, 38, 230, 0.22);
        }
        div.stButton > button:focus-visible,
        div[data-testid="stFormSubmitButton"] > button:focus-visible {
            outline: 2px solid var(--nm-cyan);
            outline-offset: 3px;
        }
            [data-testid="stForm"] {
                padding: 1rem;
                border: 1px solid rgba(0, 217, 255, 0.2);
                border-radius: 12px;
                background: rgba(11, 12, 21, 0.64);
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
                -webkit-backdrop-filter: blur(18px);
                backdrop-filter: blur(18px);
            }
            [data-testid="stMetric"] {
                padding: 0.9rem 1rem;
                border: 1px solid rgba(123, 63, 255, 0.28);
                border-radius: 10px;
                background: rgba(11, 12, 21, 0.7);
            }
            [data-testid="stMetricValue"] {
                color: var(--nm-cyan);
            }
            [data-testid="stTabs"] [role="tab"] {
                color: var(--nm-muted);
            }
            [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
                color: var(--nm-cyan);
                border-bottom-color: var(--nm-cyan);
            }
        input, textarea, [data-baseweb="select"] > div {
            color: var(--nm-text) !important;
            background: rgba(11, 12, 21, 0.82) !important;
            border-color: rgba(123, 63, 255, 0.45) !important;
            border-radius: 9px !important;
        }
        [data-testid="stSelectbox"] [role="group"] {
            background: rgba(11, 12, 21, 0.82) !important;
            border: 1px solid rgba(123, 63, 255, 0.45) !important;
            border-radius: 9px !important;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
        }
        [data-testid="stSelectbox"] [role="group"] button {
            color: var(--nm-cyan) !important;
            background: transparent !important;
            border: 0 !important;
        }
        [data-testid="stSelectbox"] [role="group"] input {
            background: transparent !important;
            border: 0 !important;
        }
        [role="listbox"] {
            color: var(--nm-text) !important;
            background: #101321 !important;
            border: 1px solid rgba(0, 217, 255, 0.45) !important;
            box-shadow: 0 0 22px rgba(40, 117, 255, 0.24);
        }
        [role="option"] {
            color: var(--nm-text) !important;
        }
        [role="option"][aria-selected="true"],
        [role="option"]:hover {
            background: rgba(40, 117, 255, 0.2) !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            border: 1px dashed rgba(0, 217, 255, 0.45);
            border-radius: 12px;
            background: rgba(11, 12, 21, 0.68);
        }
        [data-testid="stSlider"] [role="slider"] {
            border-color: var(--nm-cyan);
            box-shadow: 0 0 12px rgba(0, 217, 255, 0.4);
        }
        [data-testid="stAlert"] {
            color: var(--nm-text);
            background: rgba(18, 20, 35, 0.82);
            border: 1px solid rgba(40, 117, 255, 0.4);
            -webkit-backdrop-filter: blur(16px);
            backdrop-filter: blur(16px);
        }
        .offline-status {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            width: fit-content;
            margin: 0 0 1.25rem;
            padding: 0.65rem 0.9rem;
            border: 1px solid rgba(0, 217, 255, 0.35);
            border-radius: 10px;
            color: var(--nm-text);
            background: rgba(11, 12, 21, 0.78);
            box-shadow: 0 0 18px rgba(0, 217, 255, 0.1);
            -webkit-backdrop-filter: blur(16px);
            backdrop-filter: blur(16px);
            font-size: 0.9rem;
            font-weight: 600;
        }
        .offline-status-dot {
            width: 0.55rem;
            height: 0.55rem;
            flex: 0 0 auto;
            border-radius: 50%;
            background: #35E39A;
            box-shadow: 0 0 10px rgba(53, 227, 154, 0.75);
        }
        .offline-mode {
            margin-left: 0.4rem;
            color: var(--nm-cyan);
            font-size: 0.68rem;
            font-weight: 700;
        }
        .study-card {
            height: 340px;
            margin: 0 0 1rem;
            padding: 1px;
            border-radius: 22px;
            perspective: 1400px;
            background: linear-gradient(120deg, var(--nm-cyan), var(--nm-blue), var(--nm-violet), var(--nm-magenta));
            box-shadow: 0 28px 64px rgba(0, 0, 0, 0.66), 0 0 28px rgba(40, 117, 255, 0.18), 0 0 42px rgba(226, 38, 230, 0.12);
        }
        .study-card-inner {
            position: relative;
            width: 100%;
            height: 100%;
            border-radius: 21px;
            transform-style: preserve-3d;
            transition: transform 650ms cubic-bezier(0.2, 0.75, 0.25, 1);
        }
        .study-card-inner.is-flipped {
            transform: rotateY(180deg);
        }
        .study-card-face {
            position: absolute;
            inset: 1px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-sizing: border-box;
            padding: 2rem;
            overflow-y: auto;
            border: 1px solid rgba(149, 177, 255, 0.24);
            border-radius: 20px;
            color: var(--nm-text);
            text-align: center;
            backface-visibility: hidden;
            -webkit-backface-visibility: hidden;
            background: linear-gradient(135deg, rgba(0, 217, 255, 0.08), transparent 48%), linear-gradient(145deg, rgba(20, 23, 42, 0.96), rgba(8, 9, 17, 0.96));
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.12), inset 0 -24px 52px rgba(0, 0, 0, 0.18);
            -webkit-backdrop-filter: blur(24px);
            backdrop-filter: blur(24px);
        }
        .study-card-back {
            transform: rotateY(180deg);
            background: linear-gradient(135deg, rgba(226, 38, 230, 0.1), transparent 48%), linear-gradient(145deg, rgba(24, 20, 43, 0.96), rgba(8, 9, 17, 0.96));
        }
        .study-card-label {
            margin-bottom: 1.25rem;
            color: var(--nm-cyan);
            font: 700 0.8rem/1.2 'Segoe UI', sans-serif;
            text-transform: uppercase;
        }
        .study-card-back .study-card-label {
            color: #F16AF0;
        }
        .study-card-copy {
            width: 100%;
            max-width: 44rem;
            min-width: 0;
            color: var(--nm-text);
            white-space: normal;
            font: 600 1.65rem/1.45 'Segoe UI', 'Trebuchet MS', sans-serif;
            overflow-wrap: anywhere;
            text-shadow: 0 0 22px rgba(0, 217, 255, 0.12);
        }
        @media (max-width: 640px) {
            .block-container { padding-top: 1.4rem; }
            .brand-title { font-size: 2.1rem; }
            .study-card { height: 300px; }
            .study-card-face { padding: 1.25rem; }
            .study-card-copy { font-size: 1.3rem; }
        }
        @media (prefers-reduced-motion: reduce) {
            .study-card-inner, div.stButton > button { transition: none; }
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <header class="brand-lockup">
        <p class="brand-eyebrow">Study Engine</p>
        <h1 class="brand-title">Nomixeo Study App</h1>
    </header>
    """,
    unsafe_allow_html=True
)

def switch_deck() -> None:
    st.session_state.active_deck = st.session_state.deck_selector


initialize_decks()

# Sidebar
st.sidebar.header("Workspace")
generation_engine = st.sidebar.radio(
    "Generation engine",
    [LOCAL_ENGINE, GEMINI_ENGINE],
    key="generation_engine"
)
api_key = ""
if generation_engine == GEMINI_ENGINE:
    api_key = st.sidebar.text_input("🔑 Gemini API Key", type="password")
    st.sidebar.markdown("[Get a free key from Google AI Studio](https://aistudio.google.com/app/apikey)")
else:
    with st.sidebar.expander(
        "Offline Model Manager",
        expanded=not LOCAL_MODEL_PATH.is_file()
    ):
        if LOCAL_MODEL_PATH.is_file():
            st.success(f"Local model is ready: {LOCAL_MODEL_FILENAME}")
        else:
            st.warning(
                "The local Llama model is missing. Download the approximately 1.93 GB model to enable offline generation."
            )
            if st.button("⬇️ Download Local AI Model (~2GB)", use_container_width=True):
                progress_bar = st.progress(0.0, text="Preparing model download...")
                status_text = st.empty()
                try:
                    download_local_model(progress_bar, status_text)
                    st.rerun()
                except Exception as error:
                    st.error(f"Model download failed: {error}")

st.session_state.deck_selector = st.session_state.active_deck
st.sidebar.selectbox(
    "Active deck",
    list(st.session_state.decks),
    key="deck_selector",
    on_change=switch_deck
)

with st.sidebar.form("new_deck_form", clear_on_submit=True):
    st.text_input("New deck name", key="new_deck_name", max_chars=40)
    st.form_submit_button("Create deck", on_click=create_deck, use_container_width=True)

if st.session_state.deck_notice:
    st.sidebar.caption(st.session_state.deck_notice)
    st.session_state.deck_notice = ""

if st.sidebar.button("Load Test Flashcards", use_container_width=True):
    load_mock_deck()

st.sidebar.button(
    "🗑️ Clear Entire Deck",
    on_click=clear_active_deck,
    use_container_width=True,
    disabled=not st.session_state.decks[st.session_state.active_deck]
)

model_path = str(MODEL_PATHS[0]) if MODEL_PATHS else None
model_ready = model_path is not None and Llama is not None
generation_ready = generation_engine == GEMINI_ENGINE or model_ready
engine_status = (
    "GEMINI CLOUD" if generation_engine == GEMINI_ENGINE
    else "LOCAL MODEL" if model_ready
    else "MOCK MODE"
)

st.markdown(
    '<div class="offline-status">'
    '<span class="offline-status-dot"></span>'
    'Nomixeo Generation Engine'
    f'<span class="offline-mode">{engine_status}</span>'
    '</div>',
    unsafe_allow_html=True
)

if generation_engine == LOCAL_ENGINE and not model_ready:
    if model_path is None:
        st.warning(
            "No local GGUF model was found. The mock study deck is loaded, "
            "so the study controls are ready while you set up local generation."
        )
        st.markdown(
            "Place a compatible `.gguf` model in the app folder or in `models/`. "
            "Then install `llama-cpp-python` from `requirements.txt` and restart the app."
        )
    else:
        st.warning(
            f"Found `{Path(model_path).name}`, but `llama-cpp-python` is not installed. "
            "Install the requirements and restart the app; the mock deck is active meanwhile."
        )

tesseract_path = Path(pytesseract.pytesseract.tesseract_cmd)
tesseract_available = tesseract_path.is_file()

if generation_engine == LOCAL_ENGINE:
    with st.expander("Offline OCR setup"):
        if not tesseract_available:
            st.write(
                f"Tesseract OCR was not found at `{tesseract_path}`. "
                "Install Tesseract at that location, then restart the app. "
                "Image text extraction runs locally and does not use a network service."
            )
        else:
            st.write("Tesseract OCR is available locally.")

# Local source extraction
st.header("Upload Study Material")
uploaded_file = st.file_uploader(
    "Upload a PDF or image",
    type=["pdf", "png", "jpg", "jpeg", "txt"]
)
if generation_engine == LOCAL_ENGINE:
    st.info(
        "⚠️ Note: Local image-to-text processing requires Tesseract OCR installed on your system path."
    )

source_text = ""
image_bytes = None
if uploaded_file is not None:
    uploaded_bytes = uploaded_file.getvalue()
    try:
        if uploaded_file.type == "application/pdf":
            source_text = extract_pdf_text(uploaded_bytes)
        elif uploaded_file.type == "text/plain" or uploaded_file.name.lower().endswith(".txt"):
            source_text = uploaded_bytes.decode("utf-8-sig", errors="replace").strip()
        else:
            st.image(uploaded_bytes, caption="Uploaded image")
            if generation_engine == GEMINI_ENGINE:
                image_bytes = uploaded_bytes
            else:
                source_text = extract_image_text(uploaded_bytes)
    except Exception as error:
        st.error(f"Study material extraction failed: {error}")

    if generation_engine == LOCAL_ENGINE and uploaded_file.type not in {"application/pdf", "text/plain"}:
        if source_text.strip():
            st.caption("Image text was extracted locally with Tesseract OCR.")
        elif tesseract_available:
            st.warning("No readable text was found in this image.")

    if source_text.strip():
        with st.expander("Preview Extracted Text"):
            st.text_area("Extracted text", source_text, height=220, disabled=True)
    elif uploaded_file.type == "application/pdf" and image_bytes is None:
        st.warning("No readable text was found in this PDF.")

card_count = st.slider(
    "Number of flashcards",
    min_value=3,
    max_value=25,
    value=10
)
difficulty = st.selectbox(
    "Difficulty",
    ["Easy", "Medium", "Hard"]
)

if st.button("Generate Flashcards", disabled=not generation_ready):
    if generation_engine == GEMINI_ENGINE and not api_key.strip():
        st.warning("Enter your Gemini API key in the sidebar to generate flashcards.")
    elif not source_text.strip() and image_bytes is None:
        st.warning("Upload a PDF, image, or text file with study material before generating cards.")
    else:
        try:
            if generation_engine == GEMINI_ENGINE:
                with st.spinner("Generating flashcards with Gemini..."):
                    generated_cards = generate_gemini_flashcards(
                        api_key.strip(),
                        source_text,
                        card_count,
                        difficulty,
                        image_bytes=image_bytes
                    )
            else:
                with st.spinner("Generating flashcards with the local model..."):
                    generated_cards = generate_local_flashcards(
                        model_path or "",
                        source_text,
                        card_count,
                        difficulty
                    )
            deck_name = st.session_state.active_deck
            st.session_state.decks[deck_name].extend(generated_cards)
            st.session_state.deck_positions[deck_name] = len(st.session_state.decks[deck_name]) - len(generated_cards)
            st.session_state.deck_show_answer[deck_name] = False
        except Exception as error:
            st.error(f"Flashcard generation failed: {error}")


# Study and Manual Creation
study_tab, manual_tab = st.tabs(["Study", "Manual Creation Mode"])

with study_tab:
    deck_name = st.session_state.active_deck
    cards = st.session_state.decks[deck_name]
    st.subheader(deck_name)

    if cards:
        current_card = st.session_state.deck_positions[deck_name]
        card = cards[current_card]
        show_answer = st.session_state.deck_show_answer[deck_name]

        st.caption(f"Card {current_card + 1} of {len(cards)}")
        question = html.escape(str(card["question"]))
        answer = html.escape(str(card["answer"]))
        flipped_class = "is-flipped" if show_answer else ""

        st.markdown(
            f"""
            <div class="study-card">
                <div class="study-card-inner {flipped_class}">
                    <div class="study-card-face study-card-front">
                        <div class="study-card-label">Question</div>
                        <div class="study-card-copy">{question}</div>
                    </div>
                    <div class="study-card-face study-card-back">
                        <div class="study-card-label">Answer</div>
                        <div class="study-card-copy">{answer}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        flip_column, previous_column, next_column, delete_column = st.columns([1, 1, 1, 1.5])
        with flip_column:
            st.button(
                "Flip Card",
                on_click=flip_card,
                use_container_width=True,
                disabled=not cards
            )
        with previous_column:
            st.button(
                "Previous",
                on_click=move_card,
                args=(-1,),
                use_container_width=True,
                disabled=current_card == 0
            )
        with next_column:
            st.button(
                "Next",
                on_click=move_card,
                args=(1,),
                use_container_width=True,
                disabled=current_card >= len(cards) - 1
            )
        with delete_column:
            st.button(
                "❌ Delete This Card",
                on_click=delete_current_card,
                use_container_width=True
            )
    else:
        st.info(f"{deck_name} is empty. Add a card in Manual Creation Mode or load the test deck.")

with manual_tab:
    st.subheader("Create a flashcard")
    st.caption(f"New cards will be added to {st.session_state.active_deck}.")
    with st.form("manual_card_form", clear_on_submit=True):
        st.text_input("Front (Question/Term)", key="manual_front", max_chars=500)
        st.text_area("Back (Answer/Definition)", key="manual_back", height=150)
        st.form_submit_button("Add Flashcard", on_click=add_manual_card, use_container_width=True)

    if st.session_state.get("manual_notice"):
        st.caption(st.session_state.manual_notice)
        st.session_state.manual_notice = ""

    st.metric("Custom cards added this session", st.session_state.custom_cards_added)