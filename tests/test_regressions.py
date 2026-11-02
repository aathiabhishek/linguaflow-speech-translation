import io
import tempfile
import unittest
import wave
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.text_utils import split_text, validate_languages
from src.evaluation_utils import load_pairs
from src.translator import Translator
from src.text_to_speech import TextToSpeech


class ChunkingTests(unittest.TestCase):
    def test_short_input_preserves_formatting(self):
        self.assertEqual(split_text("Hello.\nWorld", len, 50), ["Hello.\nWorld"])

    def test_long_text_keeps_every_word_and_respects_limit(self):
        text = "Hello world. Another sentence follows! More words here."
        chunks = split_text(text, len, 18)
        self.assertTrue(all(len(c) <= 18 for c in chunks))
        self.assertEqual(" ".join(chunks).split(), text.split())

    def test_long_sentence_split(self):
        text = "one two three four five six seven"
        chunks = split_text(text, lambda x: len(x.split()) + 2, 5)
        self.assertTrue(all(len(x.split()) + 2 <= 5 for x in chunks))
        self.assertEqual(" ".join(chunks), text)

    def test_oversized_word_is_rejected(self):
        with self.assertRaises(ValueError):
            split_text("x" * 100, len, 20)

    def test_invalid_same_language_is_rejected(self):
        with self.assertRaises(ValueError):
            Translator().translate("Hello", "xx", "xx")

    def test_empty_and_passthrough_need_no_model(self):
        t = Translator()
        self.assertEqual(t.translate("", "fr", "en"), "")
        self.assertEqual(t.translate("same", "en", "en"), "same")
        self.assertEqual(t.models, {})


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    model_max_length = 512
    eos_token_id = 2
    src_lang = None
    def encode(self, text, **kwargs):
        return list(range(len(text.split()) + 2))
    def __call__(self, text, **kwargs):
        return FakeBatch(input_ids=text)
    def get_lang_id(self, language):
        return {"en": 101, "fr": 102}[language]
    def decode(self, ids, **kwargs):
        return "translated"


class FakeIds(list):
    def __getitem__(self, index):
        value = super().__getitem__(index)
        return SimpleNamespace(item=lambda: value)


class FakeModel:
    config = SimpleNamespace(max_position_embeddings=512)
    generation_config = SimpleNamespace(eos_token_id=2)
    def __init__(self):
        self.calls = []
    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [FakeIds([0, 101, 10, 2])]


class TranslatorTests(unittest.TestCase):
    def setUp(self):
        self.t = Translator()
        self.tokenizer = FakeTokenizer()
        self.model = FakeModel()
        self.loader = patch.object(self.t, "_load_model", return_value=(
            SimpleNamespace(inference_mode=nullcontext), self.tokenizer, self.model))
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def test_target_token_and_source_language_switch_both_directions(self):
        self.t.translate("bonjour", "fr", "en")
        self.assertEqual(self.tokenizer.src_lang, "fr")
        self.assertEqual(self.model.calls[-1]["forced_bos_token_id"], 101)
        self.t.translate("hello", "en", "fr")
        self.assertEqual(self.tokenizer.src_lang, "en")
        self.assertEqual(self.model.calls[-1]["forced_bos_token_id"], 102)

    def test_marian_does_not_force_m2m_language_token(self):
        self.t.backend = "marian"
        self.t.translate("bonjour", "fr", "en")
        self.assertNotIn("forced_bos_token_id", self.model.calls[-1])

    def test_output_limit_is_not_silently_accepted(self):
        self.model.generate = lambda **kw: [FakeIds([10] * 512)]
        with self.assertRaisesRegex(RuntimeError, "output limit"):
            self.t.translate("bonjour", "fr", "en")

    def test_greedy_decoding_preserved(self):
        self.t.translate("bonjour", "fr", "en")
        self.assertEqual(self.model.calls[0]["num_beams"], 1)
        self.assertFalse(self.model.calls[0]["do_sample"])


class EvaluationTests(unittest.TestCase):
    def test_aligned_pairs_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            en, fr = Path(tmp)/"en.txt", Path(tmp)/"fr.txt"
            en.write_text("hello\nworld\n")
            fr.write_text("bonjour\nmonde\n")
            self.assertEqual(load_pairs(en, fr, 2)[0], ["hello", "world"])
            with self.assertRaises(ValueError): load_pairs(en, fr, 3)
            fr.write_text("bonjour\n")
            with self.assertRaises(ValueError): load_pairs(en, fr, 1)
            fr.write_text("bonjour\n\n")
            with self.assertRaises(ValueError): load_pairs(en, fr, 2)


class TTSTests(unittest.TestCase):
    def test_atomic_output_keeps_existing_file_on_synthesis_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"out.wav"
            output.write_bytes(b"existing")
            engine = TextToSpeech(tmp)
            with patch.object(engine, "_get_voice", return_value=SimpleNamespace(
                    synthesize_wav=lambda *a: (_ for _ in ()).throw(RuntimeError("failed")))):
                with self.assertRaises(RuntimeError): engine.generate_audio("hello", "en", str(output))
            self.assertEqual(output.read_bytes(), b"existing")
            self.assertEqual(list(Path(tmp).glob("*.wav")), [output])

    def test_successful_audio_is_valid_wav(self):
        def synth(text, writer):
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(22050)
            writer.writeframes(b"\0\0" * 100)
        with tempfile.TemporaryDirectory() as tmp:
            engine = TextToSpeech(tmp)
            with patch.object(engine, "_get_voice", return_value=SimpleNamespace(synthesize_wav=synth)):
                name = engine.generate_audio("hello", "en", str(Path(tmp)/"out.wav"))
            with wave.open(name) as audio: self.assertEqual(audio.getnframes(), 100)

    def test_missing_config_triggers_download_even_with_model_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = TextToSpeech(tmp)
            Path(tmp, "en_US-lessac-medium.onnx").write_bytes(b"model")
            with patch.dict("os.environ", {"HF_HUB_OFFLINE": "0"}), \
                 patch("src.text_to_speech.subprocess.run", side_effect=__import__('subprocess').TimeoutExpired("download", 300)) as run:
                with self.assertRaisesRegex(RuntimeError, "timed out"):
                    engine.generate_audio("hello", "en", str(Path(tmp)/"out.wav"))
                run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
