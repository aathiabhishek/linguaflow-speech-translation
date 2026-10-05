# 🌐 LinguaFlow — Real-Time Speech & Text Translation

> A local AI translation application that converts spoken **or typed** English ↔ French into translated text and synthesized speech.

LinguaFlow is an end-to-end translation pipeline built with **Whisper, MarianMT, Piper TTS, PyTorch, and Streamlit**.

The application is designed around **local inference**, responsive UX, and a modular architecture suitable for extending into a real-time streaming speech system.

---

## ✨ Features

* 🎙️ Speech-to-text transcription with OpenAI Whisper (Base model)
* ⌨️ Type text directly and translate it, no microphone required
* ✏️ Editable transcript: fix recognition mistakes and re-translate
* 🌍 English ↔ French neural machine translation
* 🔊 Local text-to-speech with Piper for both spoken and typed input
* 🔄 Background TTS generation, so text appears before audio
* 🔁 One-click language swap that moves the translation into the input box
* 🧹 Clear button to reset text, translation, and audio
* 📊 Per-stage latency metrics
* 🍎 Apple Silicon MPS acceleration when available
* 🧠 MarianMT and Piper models loaded lazily and cached
* 🌑 Dark-mode responsive interface
* 🛡️ Graceful handling of empty input, silent recordings, and stage failures
* 🧩 Modular speech, translation, and TTS components

---

## 🖥️ User Interface

The page is laid out top to bottom:

1. **Header**
2. **Language selection**: From / To dropdowns with a swap button
3. **🎤 Speak**: record your voice; translation starts automatically
4. **🌍 Translate**:
   * **Left**: editable text box. Type here, or see your transcribed speech.
   * **Right**: the translation
5. **🔊 Translated Speech**: audio player
6. **⚡ Performance**: Whisper, MarianMT, and Piper timings

Two ways to use it:

```text
Speak  → transcript appears in the text box → translation → audio
Type   → click Translate                    → translation → audio
```

---

## 🏗️ Architecture

```text
        ┌───────────────────┐        ┌───────────────────┐
        │   Microphone      │        │   Typed Text      │
        │  (st.audio_input) │        │   (text box)      │
        └─────────┬─────────┘        └─────────┬─────────┘
                  │                            │
                  ▼                            │
        ┌───────────────────┐                  │
        │   Whisper Base    │                  │
        │  Speech-to-Text   │                  │
        └─────────┬─────────┘                  │
                  │                            │
                  └────────────┬───────────────┘
                               │
                          Source text
                               │
                               ▼
                     ┌───────────────────┐
                     │     MarianMT      │
                     │ Neural Translation│
                     └─────────┬─────────┘
                               │
                        Translated Text
                               │
                     ┌─────────┴─────────┐
                     │                   │
                     ▼                   ▼
              Display immediately   Background task
                                         │
                                         ▼
                                 ┌───────────────────┐
                                 │      Piper        │
                                 │   Local TTS       │
                                 └─────────┬─────────┘
                                           │
                                           ▼
                                      Audio Output
```

The app runs as a small state machine stored in `st.session_state`:

```text
idle → transcribing → translating → tts → complete
          (speech only)
```

Typed input skips `transcribing` and starts at `translating`.

---

## 🧠 Technology Stack

| Layer                 | Technology               |
| --------------------- | ------------------------ |
| UI                    | Streamlit                |
| Programming           | Python                   |
| Speech-to-Text        | OpenAI Whisper Base      |
| Translation           | Helsinki-NLP MarianMT    |
| Text-to-Speech        | Piper                    |
| Deep Learning         | PyTorch                  |
| Hardware Acceleration | Apple MPS / CPU fallback |
| Audio Processing      | FFmpeg                   |
| Frontend Styling      | CSS                      |

---

## 📁 Project Structure

```text
speech-translation/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── styles/
│   └── style.css
│
├── src/
│   ├── __init__.py
│   ├── speech_to_text.py
│   ├── translator.py
│   └── text_to_speech.py
│
├── models/
│   └── voices/
│       └── .gitkeep
│
└── audio/
    └── .gitkeep
```

---

## 🔄 Processing Pipeline

### 1. Input

**Speech**: the recording is saved to `audio/input.wav` and sent to Whisper. Each recording is hashed, so the same recording is never processed twice, and recording again always works.

**Text**: the contents of the text box are sent straight to translation. Empty input shows a message instead of running the pipeline.

---

### 2. Speech-to-Text

Recorded audio is transcribed locally using Whisper **Base**, which is more reliable than Tiny for short phrases, names, and conversational speech.

```text
Speech
  ↓
Whisper Base
  ↓
Transcript → shown in the editable text box
```

If no speech is detected, the app shows a message and returns to idle.

---

### 3. Neural Translation

The text is passed to a language-specific MarianMT model.

English → French:

```text
Helsinki-NLP/opus-mt-en-fr
```

French → English:

```text
Helsinki-NLP/opus-mt-fr-en
```

Models are loaded lazily and cached after the first use. If both language dropdowns are the same, the text is passed through unchanged.

---

### 4. Text-to-Speech

The translated text is synthesized locally using Piper.

English voice:

```text
en_US-lessac-medium
```

French voice:

```text
fr_FR-mls-medium
```

The TTS runs in a background thread so the translation can be displayed without waiting for audio. The Piper model is loaded in the main script thread and passed to the worker, because Streamlit's `st.cache_resource` is unreliable inside worker threads. A `st.fragment` polls the background task every 0.5 s and refreshes the page when audio is ready.

---

## ⚡ Performance Design

LinguaFlow separates the user-facing translation result from audio synthesis.

Instead of:

```text
Input → STT → Translation → TTS → Display everything
```

LinguaFlow uses:

```text
Input → STT → Translation → Display translation → Background TTS → Audio
```

This reduces perceived latency and lets users read the translated result while the audio is still being generated.

The application exposes individual latency measurements for:

* Whisper STT (speech input only)
* MarianMT translation
* Piper TTS

---

## 🚀 Getting Started

### Prerequisites

* Python 3.11
* Conda or virtual environment
* FFmpeg
* A recent Streamlit version (**1.37+** recommended) for `st.audio_input`, `st.fragment`, and column alignment options
* macOS Apple Silicon recommended for MPS acceleration

---

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd speech-translation
```

---

### 2. Create the environment

```bash
conda create -n speech-translation python=3.11 -y
conda activate speech-translation
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Download Piper voice models

Create the voice directory:

```bash
mkdir -p models/voices
```

Download the English voice:

```bash
python -m piper.download_voices \
  --data-dir models/voices \
  en_US-lessac-medium
```

Download the French voice:

```bash
python -m piper.download_voices \
  --data-dir models/voices \
  fr_FR-mls-medium
```

Voice models are intentionally excluded from Git because of their size and individual model licensing terms.

---

### 5. Verify FFmpeg

```bash
ffmpeg -version
```

---

### 6. Start the application

```bash
streamlit run app.py
```

Open the Streamlit URL shown in the terminal.

---

## 🖥️ Hardware Acceleration

LinguaFlow detects Apple Silicon MPS automatically.

```python
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
```

This allows the same codebase to run on systems without Apple Silicon using CPU inference.

---

## 📊 Model Configuration

### Whisper

Current model:

```text
base
```

For faster but less accurate transcription use `tiny`. For higher quality use `small` or `medium`. Change it in `app.py`:

```python
SpeechToText(model_size="base")
```

---

### MarianMT

Supported directions:

```text
English → French
French → English
```

Translation models are loaded only when the corresponding direction is requested.

Decoding is currently greedy (`num_beams=1`) for speed. Raising it to `4` in `src/translator.py` usually improves quality on short sentences and unusual names, at a small latency cost.

---

### Piper

Piper voices are loaded lazily and cached in memory.

Voice mappings live in `src/text_to_speech.py`:

```python
VOICES = {
    "en": "en_US-lessac-medium",
    "fr": "fr_FR-mls-medium",
}
```

---

## 🩺 Troubleshooting

**HTML code appears on the page.**
Markdown treats lines indented by 4+ spaces as code. `app.py` renders HTML through an `html_block()` helper that strips indentation. Use it for any new `st.markdown(..., unsafe_allow_html=True)` block.

**The swap button does nothing / raises an error.**
Language dropdowns are controlled by their widget keys. Swapping must happen in an `on_click` callback, which is how `app.py` does it.

**A new recording is ignored.**
Make sure the previous run has finished. Recordings are accepted when the pipeline is `idle` or `complete`.

**The spoken audio sounds wrong or garbled.**
Piper reads exactly the text it is given, so check the text first. Common causes:

* The French voice. `fr_FR-mls-medium` can be inconsistent; `fr_FR-siwis-medium` or `fr_FR-tom-medium` are often cleaner. Download one with `piper.download_voices` and update `VOICES`.
* Unusual names. Piper guesses the pronunciation of names it doesn't know, and a different spelling can help for the audio.
* Test voices directly:

```python
from src.text_to_speech import TextToSpeech

tts = TextToSpeech("models/voices")
tts.generate_audio("Bonjour, comment allez-vous ?", "fr", "audio/test_fr.wav")
```

**First run is slow.**
Whisper, MarianMT, and Piper models are loaded on first use and cached afterwards. MarianMT models are also downloaded from Hugging Face the first time each direction is used.

---

## 🔐 Privacy

The speech recognition, translation, and text-to-speech pipeline is designed for local inference after the required models have been downloaded.

The application does not require:

* OpenAI API keys
* Google Cloud credentials
* Translation API keys
* TTS API keys
* Vector databases
* LLM API calls

Piper TTS runs locally after its voice models are installed.

---

## 🧪 Example

Spoken input:

```text
Hello, how are you today?
```

Whisper (shown in the text box, editable):

```text
Hello, how are you today?
```

MarianMT:

```text
Bonjour, comment allez-vous aujourd'hui ?
```

Piper:

```text
🔊 French synthesized speech
```

Typed input follows the same path, starting at the translation step.

---

## 🛠️ Future Improvements

Potential next-stage improvements include:

* [ ] Real-time microphone streaming
* [ ] Voice activity detection
* [ ] Streaming Whisper transcription
* [ ] Sentence-level incremental translation
* [ ] Streaming TTS playback
* [ ] Additional language pairs
* [ ] Automatic language detection UI
* [ ] Translation history
* [ ] Copy-to-clipboard button for translations
* [ ] Speaker diarization
* [ ] Docker deployment
* [ ] Automated testing
* [ ] CI/CD pipeline
* [ ] Production API using FastAPI

---

## 🎯 Engineering Highlights

This project demonstrates practical experience with:

* Speech AI
* Natural Language Processing
* Neural Machine Translation
* Text-to-Speech
* PyTorch inference
* Apple Silicon acceleration
* Model caching
* Background task execution
* State-machine driven Streamlit apps
* Latency-aware application design
* Modular Python architecture
* Responsive UI design

---

## 📌 Project Status

**Status:** Active development

The current version provides a complete English ↔ French pipeline using local AI models, accepting both spoken and typed input.

The next architectural milestone is moving from record-and-process interaction toward **near-real-time streaming speech translation**.

---

## 📄 License

This repository's application code can be licensed according to your preferred open-source or portfolio license.

Third-party models and voice checkpoints retain their respective licenses. Review the individual model cards before redistributing or using the models commercially.
