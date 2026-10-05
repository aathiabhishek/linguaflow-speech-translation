import hashlib
import html
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import streamlit as st

from src.speech_to_text import SpeechToText
from src.translator import Translator
from src.text_to_speech import TextToSpeech


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LinguaFlow",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# HELPERS
# ============================================================

def html_block(markup: str):
    """
    Render HTML without Markdown treating it as a code block.

    Markdown treats lines indented by 4+ spaces as code, and blank
    lines can end an HTML block early. Stripping each line and
    dropping empty ones avoids both problems.
    """
    cleaned = "\n".join(
        line.strip()
        for line in markup.strip().splitlines()
        if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


def safe_text(text: str) -> str:
    """Escape user/model text for HTML and keep line breaks."""
    return html.escape(text).replace("\n", "<br>")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

STYLES_DIR = BASE_DIR / "styles"

AUDIO_DIR = BASE_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

VOICE_DIR = BASE_DIR / "models" / "voices"

INPUT_AUDIO = AUDIO_DIR / "input.wav"
OUTPUT_AUDIO = AUDIO_DIR / "translated_output.wav"


# ============================================================
# LOAD CSS
# ============================================================

css_file = STYLES_DIR / "style.css"

if css_file.exists():
    st.markdown(
        f"<style>{css_file.read_text()}</style>",
        unsafe_allow_html=True,
    )


# ============================================================
# CONSTANTS
# ============================================================

LANGUAGES = {
    "English": "en",
    "French": "fr",
}

BUSY_STAGES = ("transcribing", "translating", "tts")


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "pipeline_stage": "idle",
    "input_path": None,
    # Text that was translated (typed or transcribed)
    "transcript": "",
    "translation": "",
    # Content of the editable text box (widget key)
    "input_text": "",
    # Used to push a new transcript into the text box before it renders
    "pending_input_text": None,
    # The selectbox widgets own these two values via their keys
    "source_language_select": "English",
    "target_language_select": "French",
    "source_language": "English",
    "target_language": "French",
    "detected_language": "",
    "stt_time": None,
    "translation_time": None,
    "tts_time": None,
    "tts_future": None,
    "audio_path": None,
    "last_audio_hash": None,
    "error_message": "",
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

# A transcript from the last run is applied here, BEFORE the text box
# is created (a widget's value can't be changed after it is rendered).
if st.session_state.pending_input_text is not None:
    st.session_state.input_text = st.session_state.pending_input_text
    st.session_state.pending_input_text = None


# ============================================================
# MODEL LOADERS
# ============================================================

@st.cache_resource
def load_stt():
    """
    Load Whisper Base once.

    Base is more reliable than Tiny for short speech,
    names, and normal conversational speech.
    """

    return SpeechToText(model_size="base")


@st.cache_resource
def load_translator():
    """
    Load MarianMT translation engine once.
    """

    return Translator()


@st.cache_resource
def load_tts():
    """
    Load Piper TTS once.
    """

    return TextToSpeech(voice_directory=str(VOICE_DIR))


# ============================================================
# BACKGROUND TTS EXECUTOR
# ============================================================

if "executor" not in st.session_state:
    st.session_state.executor = ThreadPoolExecutor(max_workers=1)


# ============================================================
# CALLBACKS
# (run before the script reruns, so widget keys can be changed)
# ============================================================

def reset_outputs():
    st.session_state.translation = ""
    st.session_state.audio_path = None
    st.session_state.detected_language = ""
    st.session_state.stt_time = None
    st.session_state.translation_time = None
    st.session_state.tts_time = None
    st.session_state.tts_future = None
    st.session_state.error_message = ""


def swap_languages():
    """Swap the dropdowns and move the translation into the text box."""

    src = st.session_state.source_language_select
    tgt = st.session_state.target_language_select

    st.session_state.source_language_select = tgt
    st.session_state.target_language_select = src

    # Like most translators: the result becomes the new input
    if st.session_state.translation:
        st.session_state.input_text = st.session_state.translation

    st.session_state.transcript = ""
    reset_outputs()
    st.session_state.pipeline_stage = "idle"


def start_text_translation():
    """Translate whatever is typed in the text box."""

    text = st.session_state.input_text.strip()

    if not text:
        st.session_state.error_message = (
            "Type something to translate first."
        )
        return

    reset_outputs()
    st.session_state.transcript = text
    st.session_state.pipeline_stage = "translating"


def clear_all():
    st.session_state.input_text = ""
    st.session_state.transcript = ""
    reset_outputs()
    st.session_state.pipeline_stage = "idle"


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns(
    [5, 1],
    vertical_alignment="center",
)

with header_left:
    html_block("""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
            <div style="font-size:34px;line-height:1;">🎙️</div>
            <h1 style="margin:0;padding:0;">LinguaFlow</h1>
        </div>
    """)

    html_block("""
        <p style="font-size:16px;margin-top:4px;margin-bottom:0;">
            Real-time speech translation powered by local AI.
        </p>
    """)

with header_right:
    html_block('<div class="local-badge">100% LOCAL AI</div>')


st.markdown("")


# ============================================================
# LANGUAGE SELECTION
# ============================================================

language_left, swap_col, language_right = st.columns(
    [5, 1, 5],
    vertical_alignment="bottom",
)

with language_left:
    source_language = st.selectbox(
        "From",
        list(LANGUAGES.keys()),
        key="source_language_select",
    )

with swap_col:
    st.button(
        "⇄",
        use_container_width=True,
        help="Swap languages",
        on_click=swap_languages,
    )

with language_right:
    target_language = st.selectbox(
        "To",
        list(LANGUAGES.keys()),
        key="target_language_select",
    )

# Keep the plain session values in sync with the widgets
st.session_state.source_language = source_language
st.session_state.target_language = target_language


# ============================================================
# SECTION 1 — SPEAK (TOP)
# ============================================================

st.markdown("### 🎤 Speak")

st.caption(
    f"Speak in {source_language}. "
    "Translation starts automatically when you finish recording. "
    "Prefer typing? Use the text box below."
)

audio_input = st.audio_input("Record your voice")


# ============================================================
# AUTOMATIC RECORDING HANDLING
# ============================================================

if audio_input is not None:

    audio_bytes_in = audio_input.getvalue()
    audio_hash = hashlib.md5(audio_bytes_in).hexdigest()

    is_new_recording = audio_hash != st.session_state.last_audio_hash
    can_start = st.session_state.pipeline_stage in ("idle", "complete")

    if is_new_recording and can_start:

        # Save uploaded recording
        with open(INPUT_AUDIO, "wb") as audio_file:
            audio_file.write(audio_bytes_in)

        st.session_state.last_audio_hash = audio_hash
        st.session_state.input_path = str(INPUT_AUDIO)

        # Reset previous output
        st.session_state.transcript = ""
        reset_outputs()

        # Start automatic processing
        st.session_state.pipeline_stage = "transcribing"

        st.rerun()


# ============================================================
# SECTION 2 — TRANSLATE (BELOW): TYPE OR SEE SPEECH TEXT
# ============================================================

st.markdown("---")
st.markdown("### 🌍 Translate")

source_col, target_col = st.columns(2, gap="large")

busy = st.session_state.pipeline_stage in BUSY_STAGES

with source_col:
    st.markdown(f"**{source_language}**")

    st.text_area(
        "Text to translate",
        key="input_text",
        height=170,
        placeholder=(
            f"Type in {source_language} here, "
            "or use the microphone above..."
        ),
        label_visibility="collapsed",
        disabled=busy,
    )

    btn_translate, btn_clear = st.columns([3, 1])

    with btn_translate:
        st.button(
            "Translate",
            type="primary",
            use_container_width=True,
            on_click=start_text_translation,
            disabled=busy,
        )

    with btn_clear:
        st.button(
            "Clear",
            use_container_width=True,
            on_click=clear_all,
            disabled=busy,
        )

    if st.session_state.pipeline_stage == "transcribing":
        st.caption("🎤 Listening and understanding your speech...")


with target_col:
    st.markdown(f"**{target_language}**")

    if st.session_state.translation:
        html_block(
            '<div class="translation-card translated">'
            f"{safe_text(st.session_state.translation)}"
            "</div>"
        )
    elif st.session_state.pipeline_stage == "translating":
        html_block(
            '<div class="translation-card empty">'
            "Translating..."
            "</div>"
        )
    else:
        html_block(
            '<div class="translation-card empty">'
            "Translation will appear here..."
            "</div>"
        )


# ============================================================
# STAGE 1 — SPEECH TO TEXT
# ============================================================

if st.session_state.pipeline_stage == "transcribing":

    stt = load_stt()

    source_code = LANGUAGES[st.session_state.source_language]

    start_time = time.perf_counter()

    try:
        transcript, detected_language = stt.transcribe(
            st.session_state.input_path,
            source_code,
        )

        transcript = (transcript or "").strip()

        if not transcript:
            st.session_state.pipeline_stage = "idle"
            st.session_state.error_message = (
                "No speech detected. Please try recording again."
            )
            st.rerun()

        st.session_state.transcript = transcript
        st.session_state.detected_language = detected_language
        st.session_state.stt_time = time.perf_counter() - start_time

        # Show the transcript in the (editable) text box on next run
        st.session_state.pending_input_text = transcript

        # Move automatically to translation
        st.session_state.pipeline_stage = "translating"

        # Rerun so the transcript becomes visible
        # before translation starts.
        st.rerun()

    except Exception as e:
        st.session_state.pipeline_stage = "idle"
        st.error(f"Speech recognition failed: {e}")


# ============================================================
# STAGE 2 — TRANSLATION
# ============================================================

def generate_tts(tts, text, language, output_path):
    """Runs in a background thread. The model is loaded by the caller."""

    tts_start = time.perf_counter()

    result = tts.generate_audio(
        text=text,
        language=language,
        output_path=str(output_path),
    )

    return result, time.perf_counter() - tts_start


if st.session_state.pipeline_stage == "translating":

    translator = load_translator()

    source_code = LANGUAGES[st.session_state.source_language]
    target_code = LANGUAGES[st.session_state.target_language]

    # If both languages are the same
    if source_code == target_code:

        st.session_state.translation = st.session_state.transcript
        st.session_state.translation_time = 0.0

    else:

        start_time = time.perf_counter()

        try:
            translated_text = translator.translate(
                st.session_state.transcript,
                source_code,
                target_code,
            )

            st.session_state.translation = translated_text
            st.session_state.translation_time = (
                time.perf_counter() - start_time
            )

        except Exception as e:
            st.session_state.pipeline_stage = "idle"
            st.error(f"Translation failed: {e}")
            st.stop()

    # --------------------------------------------------------
    # Start Piper TTS in background
    # --------------------------------------------------------

    try:
        # Load in the main script thread (st.cache_resource is
        # not reliable from worker threads without script context)
        tts_engine = load_tts()

        future = st.session_state.executor.submit(
            generate_tts,
            tts_engine,
            st.session_state.translation,
            target_code,
            OUTPUT_AUDIO,
        )

    except Exception as e:
        st.session_state.pipeline_stage = "idle"
        st.error(f"Could not start speech generation: {e}")
        st.stop()

    st.session_state.tts_future = future
    st.session_state.audio_path = None
    st.session_state.pipeline_stage = "tts"

    # Rerun so translation appears immediately
    st.rerun()


# ============================================================
# STAGE 3 — TTS MONITOR
# ============================================================

@st.fragment(run_every="0.5s")
def monitor_tts():

    if st.session_state.pipeline_stage != "tts":
        return

    future = st.session_state.tts_future

    if future is None or not future.done():
        return

    try:
        audio_path, tts_elapsed = future.result()

        st.session_state.audio_path = audio_path
        st.session_state.tts_time = tts_elapsed
        st.session_state.pipeline_stage = "complete"

    except Exception as e:
        st.session_state.pipeline_stage = "idle"
        st.session_state.error_message = (
            f"Speech generation failed: {e}"
        )

    st.session_state.tts_future = None

    # Full rerun so the audio player and status refresh
    st.rerun()


monitor_tts()


# ============================================================
# AUDIO OUTPUT
# ============================================================

if (
    st.session_state.pipeline_stage == "complete"
    and st.session_state.audio_path
):

    audio_path = Path(st.session_state.audio_path)

    if audio_path.exists():

        st.markdown("### 🔊 Translated Speech")

        st.audio(
            audio_path.read_bytes(),
            format="audio/wav",
        )


# ============================================================
# PIPELINE STATUS
# ============================================================

st.markdown("")

stage = st.session_state.pipeline_stage

if st.session_state.error_message:

    st.error(st.session_state.error_message)

elif stage == "transcribing":

    st.info("🎤 Understanding your speech...")

elif stage == "translating":

    st.info("🌍 Translating...")

elif stage == "tts":

    st.info("🔊 Generating translated audio...")

elif stage == "complete":

    st.success("✓ Translation ready")


# ============================================================
# PERFORMANCE METRICS
# ============================================================

has_metrics = any(
    value is not None
    for value in [
        st.session_state.stt_time,
        st.session_state.translation_time,
        st.session_state.tts_time,
    ]
)

if has_metrics:

    st.markdown("---")
    st.markdown("### ⚡ Performance")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        stt_time = st.session_state.stt_time
        st.metric(
            "Whisper STT",
            f"{stt_time:.2f}s" if stt_time is not None else "—",
        )

    with metric2:
        translation_time = st.session_state.translation_time
        st.metric(
            "MarianMT",
            f"{translation_time:.2f}s"
            if translation_time is not None
            else "—",
        )

    with metric3:
        tts_time = st.session_state.tts_time

        if tts_time is not None:
            tts_label = f"{tts_time:.2f}s"
        elif stage == "tts":
            tts_label = "Processing..."
        else:
            tts_label = "—"

        st.metric("Piper TTS", tts_label)


# ============================================================
# TECHNOLOGY FOOTER
# ============================================================

html_block("""
    <div style="text-align:center;padding:12px 0 4px 0;">
        <div style="color:#64748b;font-size:11px;font-weight:700;letter-spacing:1.2px;">
            WHISPER BASE • MARIANMT • PIPER
        </div>
        <div style="color:#475569;font-size:11px;margin-top:6px;">
            LOCAL INFERENCE • ENGLISH ↔ FRENCH
        </div>
    </div>
""")