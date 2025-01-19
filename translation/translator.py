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
    transcription_ready = pyqtSignal(str, str)  # (text, timestamp)
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
        self.text_buffer = []
        self.buffer_timeout = 3.0  # seconds to wait before processing buffer
        self.last_transcription_time = None

    def load_model(self):
        self.model_loaded.emit(f"Loading {self.model_type} {self.model_name} model...")
        with self.model_lock:
            if self.model_type == "whisper":
                self.model = whisper.load_model(self.model_name)
            else:  # faster-whisper
                # Use CPU compute type for better compatibility, can be changed to 'cuda' for GPU
                self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        self.model_loaded.emit(f"Model {self.model_name} loaded successfully!")
    
    def set_language(self, language):
        self.language = language
    
    def set_model(self, model_name, model_type):
        if self.model_name != model_name or self.model_type != model_type:
            self.model_name = model_name
            self.model_type = model_type
            self.load_model()
    
    def run(self):
        self.running = True
        self.load_model()
        self.last_transcription_time = None
        while self.running:
            try:
                audio_data = self.audio_queue.get(timeout=1)
                current_time = time.time()
                with self.model_lock:
                    if self.model_type == "whisper":
                        result = self.model.transcribe(audio_data, language=self.language)
                        text = result["text"].strip()
                    else:  # faster-whisper
                        # Convert float32 numpy array to the correct format
                        segments, _ = self.model.transcribe(audio_data, language=self.language)
                        text = " ".join([segment.text for segment in segments]).strip()
                    
                if text:
                    self.text_buffer.append(text)
                                        # Process buffer if timeout exceeded or buffer getting too large
                    if (self.last_transcription_time and 
                        current_time - self.last_transcription_time > self.buffer_timeout) or \
                        len(self.text_buffer) > 5:  # Adjust threshold as needed
                        
                        combined_text = " ".join(self.text_buffer)
                        self.text_buffer = []
                    # Handle translation if enabled
                        if self.auto_translate and self.translation_manager:
                            try:
                                source_lang = "English" if self.language == "en" else "Chinese"
                                target_lang = "Chinese" if self.language == "en" else "English"
                                translation = self.translation_manager.translate(combined_text, source_lang, target_lang)
                                combined_text = f"{combined_text}\n[Translation] {translation}"
                            except Exception as e:
                                combined_text = f"{combined_text}\n[Translation Error] {str(e)}"
                        
                        if self.db_manager:
                            timestamp = self.db_manager.add_transcription(combined_text)
                            self.transcription_ready.emit(combined_text, timestamp)
                        else:
                            self.transcription_ready.emit(combined_text, "")
                    self.last_transcription_time = current_time
            except queue.Empty:
                                # Process any remaining text in buffer
                if self.text_buffer:
                    combined_text = " ".join(self.text_buffer)
                    self.text_buffer = []
                    
                    if self.auto_translate and self.translation_manager:
                        try:
                            source_lang = "English" if self.language == "en" else "Chinese"
                            target_lang = "Chinese" if self.language == "en" else "English"
                            translation = self.translation_manager.translate(combined_text, source_lang, target_lang)
                            combined_text = f"{combined_text}\n[Translation] {translation}"
                        except Exception as e:
                            combined_text = f"{combined_text}\n[Translation Error] {str(e)}"
                    
                    if self.db_manager:
                        timestamp = self.db_manager.add_transcription(combined_text)
                        self.transcription_ready.emit(combined_text, timestamp)
                    else:
                        self.transcription_ready.emit(combined_text, "")
                continue
    
    def stop(self):
        self.running = False