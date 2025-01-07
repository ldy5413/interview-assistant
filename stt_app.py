import sys
import numpy as np
import sounddevice as sd
import whisper
from faster_whisper import WhisperModel
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                           QVBoxLayout, QWidget, QTextEdit, QLabel,
                           QComboBox, QHBoxLayout, QSpinBox, QFileDialog,
                           QCheckBox, QDialog, QLineEdit, QGroupBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import queue
import threading
import time
import sqlite3
from datetime import datetime
import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

CHUNK_DURATION = 3  # seconds
SAMPLE_RATE = 16000

# Define model configurations
WHISPER_MODELS = {
    "Tiny (fast, less accurate)": ("tiny", "whisper"),
    "Base": ("base", "whisper"),
    "Small": ("small", "whisper"),
    "Medium": ("medium", "whisper"),
    "Large (slow, most accurate)": ("large", "whisper"),
    "Faster Tiny": ("tiny", "faster"),
    "Faster Base": ("base", "faster"),
    "Faster Small": ("small", "faster"),
    "Faster Medium": ("medium", "faster"),
    "Faster Large": ("large", "faster")
}

class APIConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("API Configuration")
        layout = QVBoxLayout(self)
        
        # Base URL
        base_url_layout = QHBoxLayout()
        base_url_layout.addWidget(QLabel("Base URL:"))
        self.base_url_input = QLineEdit()
        default_base_url = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
        self.base_url_input.setPlaceholderText(default_base_url)
        base_url_layout.addWidget(self.base_url_input)
        layout.addLayout(base_url_layout)
        
        # API Key
        api_key_layout = QHBoxLayout()
        api_key_layout.addWidget(QLabel("API Key:"))
        self.api_key_input = QLineEdit()
        default_api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
        self.api_key_input.setPlaceholderText(default_api_key)
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        api_key_layout.addWidget(self.api_key_input)
        layout.addLayout(api_key_layout)
        
        # Model Name
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model Name:"))
        self.model_input = QLineEdit()
        default_model = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
        self.model_input.setPlaceholderText(default_model)
        model_layout.addWidget(self.model_input)
        layout.addLayout(model_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

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

class DatabaseManager:
    def __init__(self):
        self.db_path = 'transcriptions.db'
        # Create the table in the main thread
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transcriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
        finally:
            conn.close()
    
    def add_transcription(self, text):
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO transcriptions (text, timestamp) VALUES (?, datetime("now", "localtime"))', (text,))
            conn.commit()
            # Get the timestamp of the just-inserted row
            cursor.execute('SELECT timestamp FROM transcriptions WHERE id = last_insert_rowid()')
            timestamp = cursor.fetchone()[0]
            return timestamp
        finally:
            conn.close()
    
    def get_last_n_lines(self, n):
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT timestamp, text FROM transcriptions ORDER BY id DESC LIMIT ?', (n,))
            return [(row[0], row[1]) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def close(self):
        # No need to close a persistent connection anymore
        pass

class AudioStreamer(QThread):
    chunk_ready = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.running = False
        
    def run(self):
        self.running = True
        try:
            device_id = sd.default.device[0]  # Use default input device
            with sd.InputStream(device=device_id,
                              samplerate=SAMPLE_RATE,
                              channels=1,
                              dtype=np.float32) as stream:
                while self.running:
                    audio_chunk, _ = stream.read(SAMPLE_RATE * CHUNK_DURATION)
                    self.chunk_ready.emit(audio_chunk.flatten())
        except Exception as e:
            print(f"Error in AudioStreamer: {str(e)}")
            self.running = False
    
    def stop(self):
        self.running = False

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
        
        while self.running:
            try:
                audio_data = self.audio_queue.get(timeout=1)
                with self.model_lock:
                    if self.model_type == "whisper":
                        result = self.model.transcribe(audio_data, language=self.language)
                        text = result["text"].strip()
                    else:  # faster-whisper
                        # Convert float32 numpy array to the correct format
                        segments, _ = self.model.transcribe(audio_data, language=self.language)
                        text = " ".join([segment.text for segment in segments]).strip()
                    
                if text:
                    # Handle translation if enabled
                    if self.auto_translate and self.translation_manager:
                        try:
                            source_lang = "English" if self.language == "en" else "Chinese"
                            target_lang = "Chinese" if self.language == "en" else "English"
                            translation = self.translation_manager.translate(text, source_lang, target_lang)
                            text = f"{text}\n[Translation] {translation}"
                        except Exception as e:
                            text = f"{text}\n[Translation Error] {str(e)}"
                    
                    if self.db_manager:
                        timestamp = self.db_manager.add_transcription(text)
                        self.transcription_ready.emit(text, timestamp)
                    else:
                        self.transcription_ready.emit(text, "")
            except queue.Empty:
                continue
    
    def stop(self):
        self.running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech to Text Converter")
        self.setMinimumSize(1000, 600)
        
        # Initialize managers
        self.db_manager = DatabaseManager()
        self.translation_manager = TranslationManager()
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create control panel
        control_panel = QHBoxLayout()
        
        # Create language selection
        self.language_combo = QComboBox()
        self.language_combo.addItems(["English", "Chinese"])
        control_panel.addWidget(QLabel("Language:"))
        control_panel.addWidget(self.language_combo)
        self.language_combo.currentTextChanged.connect(self.language_changed)
        
        # Create model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(WHISPER_MODELS.keys())
        control_panel.addWidget(QLabel("Model:"))
        control_panel.addWidget(self.model_combo)
        self.model_combo.currentTextChanged.connect(self.model_changed)
        
        # Create translation checkbox and config button
        self.translate_check = QCheckBox("Auto Translate")
        self.api_config_button = QPushButton("API Settings")
        control_panel.addWidget(self.translate_check)
        control_panel.addWidget(self.api_config_button)
        self.api_config_button.clicked.connect(self.configure_api)
        
        # Create start/stop button
        self.toggle_button = QPushButton("Start Streaming")
        control_panel.addWidget(self.toggle_button)
        
        # Add control panel to main layout
        layout.addLayout(control_panel)
        
        # Add status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)
        
        # Create text output containers
        text_container = QHBoxLayout()
        
        # Original text section
        original_section = QVBoxLayout()
        original_section.addWidget(QLabel("Original Transcription:"))
        self.original_output = QTextEdit()
        self.original_output.setReadOnly(True)
        original_section.addWidget(self.original_output)
        text_container.addLayout(original_section)
        
        # Translation section
        translation_section = QVBoxLayout()
        translation_section.addWidget(QLabel("Translation:"))
        self.translation_output = QTextEdit()
        self.translation_output.setReadOnly(True)
        translation_section.addWidget(self.translation_output)
        text_container.addLayout(translation_section)
        
        # Add text container to main layout
        layout.addLayout(text_container)
        
        # Create export panel
        export_panel = QHBoxLayout()
        self.line_count_spinbox = QSpinBox()
        self.line_count_spinbox.setRange(1, 1000)
        self.line_count_spinbox.setValue(1000)
        self.export_button = QPushButton("Export")
        self.load_history_button = QPushButton("Load History")
        
        export_panel.addWidget(QLabel("Number of lines:"))
        export_panel.addWidget(self.line_count_spinbox)
        export_panel.addWidget(self.export_button)
        export_panel.addWidget(self.load_history_button)
        export_panel.addStretch()
        
        # Add export panel to main layout
        layout.addLayout(export_panel)
        
        # Connect signals
        self.toggle_button.clicked.connect(self.toggle_streaming)
        self.export_button.clicked.connect(self.export_transcription)
        self.load_history_button.clicked.connect(self.load_history)
        
        # Initialize threads
        self.audio_streamer = None
        self.transcriber = None
        self.is_streaming = False
        
    def language_changed(self, new_language):
        if self.transcriber:
            language_codes = {"English": "en", "Chinese": "zh"}
            self.transcriber.set_language(language_codes[new_language])
    
    def model_changed(self, new_model):
        if self.transcriber:
            model_name, model_type = WHISPER_MODELS[new_model]
            self.transcriber.set_model(model_name, model_type)
    
    def toggle_streaming(self):
        if not self.is_streaming:
            self.start_streaming()
        else:
            self.stop_streaming()
    
    def start_streaming(self):
        language_codes = {"English": "en", "Chinese": "zh"}
        current_language = language_codes[self.language_combo.currentText()]
        model_name, model_type = WHISPER_MODELS[self.model_combo.currentText()]
        
        if self.translate_check.isChecked() and not self.translation_manager.api_key:
            self.status_label.setText("Please configure API settings for translation")
            return
        
        self.transcriber = TranscriptionThread(
            language=current_language, 
            model_name=model_name,
            model_type=model_type,
            db_manager=self.db_manager,
            translation_manager=self.translation_manager,
            auto_translate=self.translate_check.isChecked()
        )
        self.transcriber.transcription_ready.connect(self.handle_transcription)
        self.transcriber.model_loaded.connect(self.update_status)
        
        self.audio_streamer = AudioStreamer()
        self.audio_streamer.chunk_ready.connect(self.handle_audio_chunk)
        
        self.audio_streamer.start()
        self.transcriber.start()
        
        self.is_streaming = True
        self.toggle_button.setText("Stop Streaming")
        self.model_combo.setEnabled(False)
        self.translate_check.setEnabled(False)
        self.api_config_button.setEnabled(False)
    
    def stop_streaming(self):
        if self.audio_streamer:
            self.audio_streamer.stop()
            self.audio_streamer.wait()
            self.audio_streamer.deleteLater()
            self.audio_streamer = None
            
        if self.transcriber:
            self.transcriber.stop()
            self.transcriber.wait()
            self.transcriber.deleteLater()
            self.transcriber = None
        
        self.is_streaming = False
        self.toggle_button.setText("Start Streaming")
        self.model_combo.setEnabled(True)
        self.translate_check.setEnabled(True)
        self.api_config_button.setEnabled(True)
        self.status_label.setText("Ready")
    
    def update_status(self, message):
        self.status_label.setText(message)
    
    def handle_audio_chunk(self, audio_chunk):
        if self.transcriber:
            self.transcriber.audio_queue.put(audio_chunk)
    
    def handle_transcription(self, text, timestamp):
        # Split the text into original and translation if translation exists
        parts = text.split("\n[Translation] ")
        original_text = parts[0]
        translation_text = parts[1] if len(parts) > 1 else ""
        
        # Format and display original text
        formatted_original = f"[{timestamp}] {original_text}"
        self.original_output.append(formatted_original)
        self.original_output.verticalScrollBar().setValue(
            self.original_output.verticalScrollBar().maximum()
        )
        
        # Format and display translation if it exists
        if translation_text:
            formatted_translation = f"[{timestamp}] {translation_text}"
            self.translation_output.append(formatted_translation)
            self.translation_output.verticalScrollBar().setValue(
                self.translation_output.verticalScrollBar().maximum()
            )
    
    def export_transcription(self):
        try:
            num_lines = self.line_count_spinbox.value()
            lines = self.db_manager.get_last_n_lines(num_lines)
            
            if not lines:
                self.status_label.setText("No transcriptions to export")
                return
            
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Save Transcription", 
                f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "Text Files (*.txt)"
            )
            
            if file_name:
                with open(file_name, 'w', encoding='utf-8') as f:
                    for timestamp, text in reversed(lines):
                        parts = text.split("\n[Translation] ")
                        original_text = parts[0]
                        translation_text = parts[1] if len(parts) > 1 else ""
                        
                        f.write(f"[{timestamp}] Original: {original_text}\n")
                        if translation_text:
                            f.write(f"[{timestamp}] Translation: {translation_text}\n")
                        f.write("\n")
                self.status_label.setText(f"Successfully exported to {file_name}")
        except Exception as e:
            self.status_label.setText(f"Error exporting: {str(e)}")
    
    def load_history(self):
        try:
            num_lines = self.line_count_spinbox.value()
            lines = self.db_manager.get_last_n_lines(num_lines)
            
            if not lines:
                self.status_label.setText("No history found in database")
                return
            
            # Clear both text outputs
            self.original_output.clear()
            self.translation_output.clear()
            
            # Add historical entries in chronological order
            for timestamp, text in reversed(lines):
                parts = text.split("\n[Translation] ")
                original_text = parts[0]
                translation_text = parts[1] if len(parts) > 1 else ""
                
                self.original_output.append(f"[{timestamp}] {original_text}")
                if translation_text:
                    self.translation_output.append(f"[{timestamp}] {translation_text}")
            
            self.status_label.setText(f"Loaded {len(lines)} historical transcriptions")
        except Exception as e:
            self.status_label.setText(f"Error loading history: {str(e)}")
    
    def closeEvent(self, event):
        if self.is_streaming:
            self.stop_streaming()
        self.db_manager.close()
        event.accept()
    
    def configure_api(self):
        dialog = APIConfigDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            base_url = dialog.base_url_input.text() or dialog.base_url_input.placeholderText()
            api_key = dialog.api_key_input.text()
            model_name = dialog.model_input.text() or dialog.model_input.placeholderText()
            if api_key:
                try:
                    self.translation_manager.configure(base_url, api_key, model_name)
                    self.status_label.setText(f"API configured successfully (Model: {model_name})")
                except Exception as e:
                    self.status_label.setText(f"API configuration failed: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 