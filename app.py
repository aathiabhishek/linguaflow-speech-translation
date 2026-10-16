import base64
import csv
import difflib
import hashlib
import html
import io
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src.speech_to_text import SpeechToText
from src.translator import Translator
from src.text_to_speech import TextToSpeech


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
    """Render HTML without Markdown treating it as code."""
    cleaned = "\n".join(
        line.strip() for line in markup.strip().splitlines() if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


def safe_text(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def tokens(text: str):
    return re.findall(r"[\w']+", text.lower().replace("’", "'"))


def compare_speech(expected: str, heard: str):
    """Word-level comparison. Returns (score 0-100, html of expected words)."""
    exp, got = tokens(expected), tokens(heard)
    sm = difflib.SequenceMatcher(None, exp, got, autojunk=False)
    parts = []
    for tag, i1, i2, _, _ in sm.get_opcodes():
        css = "w-ok" if tag == "equal" else "w-bad"
        parts += [f'<span class="{css}">{html.escape(w)}</span>' for w in exp[i1:i2]]
    matched = sum(b.size for b in sm.get_matching_blocks())
    return round(100 * matched / max(len(exp), 1)), " ".join(parts)


def audio_player(path: Path):
    """Audio player with slow-speed buttons (useful for learners)."""
    b64 = base64.b64encode(path.read_bytes()).decode()
    components.html(
        f"""
        <style>
          body{{margin:0;font-family:Inter,system-ui,sans-serif}}
          .row{{display:flex;gap:8px;margin-top:10px}}
          button{{padding:7px 14px;border-radius:999px;border:1px solid #243352;
                  background:#111a2e;color:#cbd5e1;cursor:pointer;font-size:13px}}
          button.on{{border-color:#7c8cff;color:#fff}}
        </style>
        <audio id="a" controls style="width:100%;height:44px"
               src="data:audio/wav;base64,{b64}"></audio>
        <div class="row">
          <button data-r="0.6">Slow 0.6×</button>
          <button data-r="0.8">0.8×</button>
          <button data-r="1" class="on">Normal</button>
        </div>
        <script>
          const a=document.getElementById('a');
          document.querySelectorAll('button').forEach(b=>b.onclick=()=>{{
            a.playbackRate=parseFloat(b.dataset.r); a.preservesPitch=true;
            document.querySelectorAll('button').forEach(x=>x.classList.remove('on'));
            b.classList.add('on'); a.currentTime=0; a.play();
          }});
        </script>
        """,
        height=100,
    )


# ============================================================
# PATHS (unique files per visitor, so users never overwrite each other)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
STYLES_DIR = BASE_DIR / "styles"
AUDIO_DIR = BASE_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
VOICE_DIR = BASE_DIR / "models" / "voices"

if "sid" not in st.session_state:
    st.session_state.sid = uuid.uuid4().hex[:12]
    # Remove audio files older than 1 hour
    for old in AUDIO_DIR.glob("*.wav"):
        try:
            if time.time() - old.stat().st_mtime > 3600:
                old.unlink()
        except OSError:
            pass

SID = st.session_state.sid
INPUT_AUDIO = AUDIO_DIR / f"{SID}_input.wav"
OUTPUT_AUDIO = AUDIO_DIR / f"{SID}_output.wav"
PRACTICE_AUDIO = AUDIO_DIR / f"{SID}_practice.wav"

css_file = STYLES_DIR / "style.css"
if css_file.exists():
    st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

LANGUAGES = {"English": "en", "French": "fr"}
FLAGS = {"English": "🇬🇧", "French": "🇫🇷"}
BUSY_STAGES = ("transcribing", "translating", "tts")

STARTERS = {
    "en": [
        "Where is the train station?",
        "How much does this cost?",
        "Could you speak more slowly?",
        "I would like a coffee, please.",
    ],
    "fr": [
        "Où est la gare ?",
        "Combien ça coûte ?",
        "Pouvez-vous parler plus lentement ?",
        "Je voudrais un café, s'il vous plaît.",
    ],
}


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "pipeline_stage": "idle",
    "input_path": None,
    "transcript": "",
    "translation": "",
    "input_text": "",
    "pending_input_text": None,
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
    "phrasebook": [],
    "practice_n": 0,
    "practice_hash": None,
    "practice_result": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

if st.session_state.pending_input_text is not None:
    st.session_state.input_text = st.session_state.pending_input_text
    st.session_state.pending_input_text = None


# ============================================================
# MODEL LOADERS
# ============================================================

@st.cache_resource
def load_stt():
    return SpeechToText(model_size="base")


@st.cache_resource
def load_translator():
    return Translator()


@st.cache_resource
def load_tts():
    return TextToSpeech(voice_directory=str(VOICE_DIR))


if "executor" not in st.session_state:
    st.session_state.executor = ThreadPoolExecutor(max_workers=1)


# ============================================================
# CALLBACKS
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
    st.session_state.practice_result = None
    st.session_state.practice_n += 1  # resets the practice recorder


def swap_languages():
    src = st.session_state.source_language_select
    tgt = st.session_state.target_language_select
    st.session_state.source_language_select = tgt
    st.session_state.target_language_select = src

    if st.session_state.translation:
        st.session_state.input_text = st.session_state.translation

    st.session_state.transcript = ""
    reset_outputs()
    st.session_state.pipeline_stage = "idle"


def start_text_translation():
    text = st.session_state.input_text.strip()
    if not text:
        st.session_state.error_message = "Type or record something to translate."
        return
    reset_outputs()
    st.session_state.transcript = text
    st.session_state.pipeline_stage = "translating"


def use_starter(text):
    st.session_state.input_text = text
    start_text_translation()


def clear_all():
    st.session_state.input_text = ""
    st.session_state.transcript = ""
    reset_outputs()
    st.session_state.pipeline_stage = "idle"


def save_phrase():
    item = {
        "src": st.session_state.transcript,
        "tgt": st.session_state.translation,
        "sl": st.session_state.source_language,
        "tl": st.session_state.target_language,
    }
    if item["src"] and item not in st.session_state.phrasebook:
        st.session_state.phrasebook.append(item)


def load_phrase(i):
    item = st.session_state.phrasebook[i]
    st.session_state.source_language_select = item["sl"]
    st.session_state.target_language_select = item["tl"]
    st.session_state.input_text = item["src"]
    st.session_state.transcript = ""
    reset_outputs()
    st.session_state.pipeline_stage = "idle"


def delete_phrase(i):
    st.session_state.phrasebook.pop(i)


# ============================================================
# HEADER
# ============================================================

html_block("""
    <div class="brand">
        <h1>LinguaFlow</h1>
        <span>Speak it, see it, hear it, repeat it.</span>
        <div class="pill">Open-source AI</div>
    </div>
""")

st.markdown("")

lang_l, swap_col, lang_r = st.columns([5, 1, 5], vertical_alignment="bottom")

with lang_l:
    source_language = st.selectbox(
        "I speak",
        list(LANGUAGES),
        key="source_language_select",
        format_func=lambda l: f"{FLAGS[l]} {l}",
    )

with swap_col:
    st.button("⇄", use_container_width=True, help="Swap languages", on_click=swap_languages)

with lang_r:
    target_language = st.selectbox(
        "I'm learning",
        list(LANGUAGES),
        key="target_language_select",
        format_func=lambda l: f"{FLAGS[l]} {l}",
    )

st.session_state.source_language = source_language
st.session_state.target_language = target_language
source_code = LANGUAGES[source_language]
target_code = LANGUAGES[target_language]


# ============================================================
# INPUT: SPEAK OR TYPE
# ============================================================

st.markdown("### Say or type a phrase")
st.caption(
    f"Record in {source_language} and translation starts when you stop, "
    "or type below."
)

audio_input = st.audio_input("Record your voice", label_visibility="collapsed")

if audio_input is not None:
    audio_bytes_in = audio_input.getvalue()
    audio_hash = hashlib.md5(audio_bytes_in).hexdigest()

    if (
        audio_hash != st.session_state.last_audio_hash
        and st.session_state.pipeline_stage in ("idle", "complete")
    ):
        INPUT_AUDIO.write_bytes(audio_bytes_in)
        st.session_state.last_audio_hash = audio_hash
        st.session_state.input_path = str(INPUT_AUDIO)
        st.session_state.transcript = ""
        reset_outputs()
        st.session_state.pipeline_stage = "transcribing"
        st.rerun()

busy = st.session_state.pipeline_stage in BUSY_STAGES

source_col, target_col = st.columns(2, gap="large")

with source_col:
    html_block(f'<div class="lang-label">{FLAGS[source_language]} {source_language}</div>')

    st.text_area(
        "Text to translate",
        key="input_text",
        height=170,
        placeholder=f"Type in {source_language}…",
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
        st.button("Clear", use_container_width=True, on_click=clear_all, disabled=busy)

with target_col:
    html_block(f'<div class="lang-label">{FLAGS[target_language]} {target_language}</div>')

    if st.session_state.translation:
        html_block(
            f'<div class="card done">{safe_text(st.session_state.translation)}</div>'
        )
    elif st.session_state.pipeline_stage == "translating":
        html_block('<div class="card empty">Translating…</div>')
    else:
        html_block('<div class="card empty">Your translation appears here.</div>')

# Starter phrases
with st.container(key="chips"):
    st.caption(f"Try a common {source_language} phrase")
    chip_cols = st.columns(len(STARTERS[source_code]))
    for col, phrase in zip(chip_cols, STARTERS[source_code]):
        with col:
            st.button(
                phrase,
                key=f"chip_{phrase}",
                on_click=use_starter,
                args=(phrase,),
                disabled=busy,
                use_container_width=True,
            )


# ============================================================
# PROGRESS STEPS
# ============================================================

stage = st.session_state.pipeline_stage
order = ["transcribing", "translating", "tts", "complete"]

if stage in order:
    idx = order.index(stage)
    steps = ""
    for i, label in enumerate(["Listen", "Translate", "Generate voice"]):
        css = "done" if i < idx else "active" if i == idx else ""
        mark = "✓ " if css == "done" else ""
        steps += f'<div class="step {css}">{mark}{label}</div>'
    html_block(f'<div class="steps">{steps}</div>')

if st.session_state.error_message:
    st.error(st.session_state.error_message)


# ============================================================
# STAGE 1 — SPEECH TO TEXT
# ============================================================

if stage == "transcribing":
    stt = load_stt()
    start_time = time.perf_counter()

    try:
        transcript, detected_language = stt.transcribe(
            st.session_state.input_path, source_code
        )
        transcript = (transcript or "").strip()

        if not transcript:
            st.session_state.pipeline_stage = "idle"
            st.session_state.error_message = "No speech detected. Try recording again."
            st.rerun()

        st.session_state.transcript = transcript
        st.session_state.detected_language = detected_language
        st.session_state.stt_time = time.perf_counter() - start_time
        st.session_state.pending_input_text = transcript
        st.session_state.pipeline_stage = "translating"
        st.rerun()

    except Exception as e:
        st.session_state.pipeline_stage = "idle"
        st.error(f"Speech recognition failed: {e}")


# ============================================================
# STAGE 2 — TRANSLATION + START TTS
# ============================================================

def generate_tts(tts, text, language, output_path):
    tts_start = time.perf_counter()
    result = tts.generate_audio(
        text=text, language=language, output_path=str(output_path)
    )
    return result, time.perf_counter() - tts_start


if st.session_state.pipeline_stage == "translating":
    translator = load_translator()

    if source_code == target_code:
        st.session_state.translation = st.session_state.transcript
        st.session_state.translation_time = 0.0
    else:
        start_time = time.perf_counter()
        try:
            st.session_state.translation = translator.translate(
                st.session_state.transcript, source_code, target_code
            )
            st.session_state.translation_time = time.perf_counter() - start_time
        except Exception as e:
            st.session_state.pipeline_stage = "idle"
            st.error(f"Translation failed: {e}")
            st.stop()

    try:
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
        st.session_state.error_message = f"Speech generation failed: {e}"

    st.session_state.tts_future = None
    st.rerun()


monitor_tts()


# ============================================================
# LISTEN, SAVE, PRACTISE
# ============================================================

stage = st.session_state.pipeline_stage

if st.session_state.translation and stage in ("tts", "complete"):
    st.markdown("---")
    st.markdown("### Listen and repeat")

    if stage == "complete" and st.session_state.audio_path:
        audio_path = Path(st.session_state.audio_path)
        if audio_path.exists():
            audio_player(audio_path)

            save_col, dl_col = st.columns(2)
            with save_col:
                st.button(
                    "☆ Save to phrasebook",
                    on_click=save_phrase,
                    use_container_width=True,
                )
            with dl_col:
                st.download_button(
                    "Download audio",
                    data=audio_path.read_bytes(),
                    file_name="translation.wav",
                    mime="audio/wav",
                    use_container_width=True,
                )
    else:
        st.caption("Generating voice…")

    with st.expander(f"🎯 Practice saying it in {target_language}", expanded=False):
        st.caption(
            "Listen first, then record yourself. "
            "Words in green matched; words in orange need another try."
        )

        practice_audio = st.audio_input(
            "Repeat the phrase",
            key=f"practice_{st.session_state.practice_n}",
            label_visibility="collapsed",
        )

        if practice_audio is not None:
            p_bytes = practice_audio.getvalue()
            p_hash = hashlib.md5(p_bytes).hexdigest()

            if p_hash != st.session_state.practice_hash:
                PRACTICE_AUDIO.write_bytes(p_bytes)
                st.session_state.practice_hash = p_hash
                with st.spinner("Checking your pronunciation…"):
                    try:
                        heard, _ = load_stt().transcribe(
                            str(PRACTICE_AUDIO), target_code
                        )
                        heard = (heard or "").strip()
                        score, words = compare_speech(
                            st.session_state.translation, heard
                        )
                        st.session_state.practice_result = {
                            "score": score,
                            "words": words,
                            "heard": heard,
                        }
                    except Exception as e:
                        st.session_state.practice_result = None
                        st.error(f"Could not check your recording: {e}")

        result = st.session_state.practice_result
        if result:
            html_block(
                f"""
                <div class="score">{result["score"]}%<small>match</small></div>
                <div class="words">{result["words"]}</div>
                <div class="heard">We heard: {safe_text(result["heard"] or "nothing")}</div>
                """
            )


# ============================================================
# PHRASEBOOK
# ============================================================

book = st.session_state.phrasebook

st.markdown("---")
with st.expander(f"📚 Phrasebook ({len(book)})", expanded=False):
    if not book:
        st.caption("Save translations to review them later. Saved phrases last until you close this tab.")
    else:
        for i, item in enumerate(book):
            c_text, c_use, c_del = st.columns([8, 1.4, 1])
            with c_text:
                html_block(
                    f'<div class="phrase"><b>{safe_text(item["tgt"])}</b>'
                    f'<div>{safe_text(item["src"])}</div></div>'
                )
            with c_use:
                st.button("Reuse", key=f"use_{i}", on_click=load_phrase, args=(i,), use_container_width=True)
            with c_del:
                st.button("✕", key=f"del_{i}", on_click=delete_phrase, args=(i,), use_container_width=True)

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["Front", "Back"])  # flashcard-ready (Anki, Quizlet)
        writer.writerows([(p["src"], p["tgt"]) for p in book])
        st.download_button(
            "Export as flashcards (CSV)",
            data=buf.getvalue(),
            file_name="linguaflow_phrasebook.csv",
            mime="text/csv",
        )


# ============================================================
# PERFORMANCE
# ============================================================

if any(
    v is not None
    for v in (
        st.session_state.stt_time,
        st.session_state.translation_time,
        st.session_state.tts_time,
    )
):
    with st.expander("⚡ Performance", expanded=False):
        m1, m2, m3 = st.columns(3)
        stt_t = st.session_state.stt_time
        tr_t = st.session_state.translation_time
        tts_t = st.session_state.tts_time
        m1.metric("Whisper STT", f"{stt_t:.2f}s" if stt_t is not None else "—")
        m2.metric("MarianMT", f"{tr_t:.2f}s" if tr_t is not None else "—")
        m3.metric(
            "Piper TTS",
            f"{tts_t:.2f}s" if tts_t is not None else ("…" if stage == "tts" else "—"),
        )

st.caption("Whisper Base · MarianMT · Piper — English ↔ French")