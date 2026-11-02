import sys
from pathlib import Path
from unittest.mock import patch
import wave
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest
class Translator:
    label = 'M2M100 418M'
    def translate(self, text, source, target):return 'Bonjour le monde'
class STT:
    def __init__(self, **kw):pass
class TTS:
    def __init__(self, **kw):pass
    def generate_audio(self,text,language,output_path):
        with wave.open(output_path,'wb') as f:
            f.setnchannels(1);f.setsampwidth(2);f.setframerate(22050);f.writeframes(b'\0\0'*100)
        return output_path
with patch('src.translator.Translator',Translator),patch('src.speech_to_text.SpeechToText',STT),patch('src.text_to_speech.TextToSpeech',TTS):
    at=AppTest.from_file(str(ROOT / 'app.py'),default_timeout=20).run()
    assert not at.exception,at.exception
    at.text_area[0].set_value('Hello world')
    next(b for b in at.button if b.label=='Translate').click().run()
    for _ in range(3):
        if at.session_state['pipeline_stage']=='complete':break
        at.run()
    assert not at.exception,at.exception
    assert at.session_state['translation']=='Bonjour le monde'
    assert at.session_state['pipeline_stage']=='complete',at.session_state['pipeline_stage']
    assert Path(at.session_state['audio_path']).is_file()
    next(b for b in at.button if b.label=='☆ Save to phrasebook').click().run()
    assert len(at.session_state['phrasebook'])==1
    next(b for b in at.button if b.label=='⇄').click().run()
    assert at.session_state['source_language_select']=='French'
    assert at.session_state['input_text']=='Bonjour le monde'
    assert not at.session_state['translation']
    assert not at.exception,at.exception
    print('UI smoke checks passed: initial rendering, typed translation, background audio completion, phrase saving, language swap.')
