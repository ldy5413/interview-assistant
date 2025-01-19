import os, time
import openai
from dotenv import load_dotenv
import queue
import threading
from PyQt6.QtCore import QThread, pyqtSignal
import whisper
from faster_whisper import WhisperModel

# Load environment variables from .env file
load_dotenv()

class TranslationManager:
    def __init__(self):
        # Initialize with environment variables if available
        self.base_url = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
        self.client = None
        
        # If API key is available in environment, configure the client
        if self.api_key:
            self.configure(self.base_url, self.api_key, self.model_name)
    
    def configure(self, base_url, api_key, model_name=None):
        self.base_url = base_url
        self.api_key = api_key
        if model_name:
            self.model_name = model_name
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
    
    def translate(self, text, source_lang, target_lang):
        if not self.client:
            raise Exception("API not configured. Please configure API settings first.")
        
        prompt = f"Translate the following {source_lang} text to {target_lang}:\n{text}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a professional translator."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"Translation failed: {str(e)}")

class TranscriptionThread(QThread):
    # New signal for streaming words
    word_ready = pyqtSignal(str, str, bool)  # (text, timestamp, is_translation)
    sentence_complete = pyqtSignal(bool)  # is_translation
    model_loaded = pyqtSignal(str)
    
    def __init__(self, language="en", model_name="base", model_type="whisper", 
                 db_manager=None, translation_manager=None, auto_translate=False):
        super().__init__()
        self.model = None
        self.audio_queue = queue.Queue()
        self.running = False
        self.language = language
        self.model_name = model_name
        self.model_type = model_type
        self.model_lock = threading.Lock()
        self.db_manager = db_manager
        self.translation_manager = translation_manager
        self.auto_translate = auto_translate
        
        # Buffers for current sentence
        self.current_sentence = []
        self.current_translation = []
        self.sentence_end_markers = {'.', '!', '?', '。', '！', '？'}
        self.last_timestamp = None

    def load_model(self):
        self.model_loaded.emit(f"Loading {self.model_type} {self.model_name} model...")
        with self.model_lock:
            if self.model_type == "whisper":
                self.model = whisper.load_model(self.model_name)
            else:  # faster-whisper
                self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        self.model_loaded.emit(f"Model {self.model_name} loaded successfully!")
    
    def set_language(self, language):
        self.language = language
    
    def set_model(self, model_name, model_type):
        if self.model_name != model_name or self.model_type != model_type:
            self.model_name = model_name
            self.model_type = model_type
            self.load_model()

    def process_text(self, text, timestamp):
        # Split text into words while preserving punctuation
        words = text.replace('.', ' .').replace('!', ' !').replace('?', ' ?')\
                   .replace('。', ' 。').replace('！', ' ！').replace('？', ' ？').split()
        
        for word in words:
            # Check if this word ends with a sentence marker
            ends_sentence = any(word.endswith(marker) for marker in self.sentence_end_markers)
            
            # Add word to current sentence
            self.current_sentence.append(word)
            
            # Emit the word
            self.word_ready.emit(word, timestamp, False)
            
            if ends_sentence:
                # Get the complete sentence
                sentence = ' '.join(self.current_sentence)
                
                # Handle translation if enabled
                if self.auto_translate and self.translation_manager:
                    try:
                        source_lang = "English" if self.language == "en" else "Chinese"
                        target_lang = "Chinese" if self.language == "en" else "English"
                        translation = self.translation_manager.translate(sentence, source_lang, target_lang)
                        
                        # Emit translation words
                        translation_words = translation.split()
                        for trans_word in translation_words:
                            self.word_ready.emit(trans_word, timestamp, True)
                        self.sentence_complete.emit(True)
                        
                    except Exception as e:
                        self.word_ready.emit(f"[Translation Error: {str(e)}]", timestamp, True)
                        self.sentence_complete.emit(True)
                
                # Store in database
                if self.db_manager:
                    self.db_manager.add_transcription(sentence)
                
                # Signal sentence completion and reset buffer
                self.sentence_complete.emit(False)
                self.current_sentence = []
    
    def run(self):
        self.running = True
        self.load_model()
        
        while self.running:
            try:
                audio_data = self.audio_queue.get(timeout=1)
                timestamp = time.strftime("%H:%M:%S")
                
                with self.model_lock:
                    if self.model_type == "whisper":
                        result = self.model.transcribe(audio_data, language=self.language)
                        text = result["text"].strip()
                    else:  # faster-whisper
                        segments, _ = self.model.transcribe(audio_data, language=self.language)
                        text = " ".join([segment.text for segment in segments]).strip()
                
                if text:
                    self.process_text(text, timestamp)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in transcription: {str(e)}")
    
    def stop(self):
        self.running = False
        # Process any remaining text in the current sentence
        if self.current_sentence:
            sentence = ' '.join(self.current_sentence)
            timestamp = time.strftime("%H:%M:%S")
            
            if self.auto_translate and self.translation_manager:
                try:
                    source_lang = "English" if self.language == "en" else "Chinese"
                    target_lang = "Chinese" if self.language == "en" else "English"
                    translation = self.translation_manager.translate(sentence, source_lang, target_lang)
                    for word in translation.split():
                        self.word_ready.emit(word, timestamp, True)
                    self.sentence_complete.emit(True)
                except Exception as e:
                    self.word_ready.emit(f"[Translation Error: {str(e)}]", timestamp, True)
                    self.sentence_complete.emit(True)
            
            if self.db_manager:
                self.db_manager.add_transcription(sentence)
            
            self.sentence_complete.emit(False)