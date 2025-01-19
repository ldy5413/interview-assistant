import sys, os
import numpy as np
import sounddevice as sd

from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                           QVBoxLayout, QWidget, QTextEdit, QLabel,
                           QComboBox, QHBoxLayout, QSpinBox, QFileDialog,
                           QCheckBox, QDialog, QLineEdit, QGroupBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from datetime import datetime
from db import DatabaseManager
from translation import TranscriptionThread, TranslationManager, AIAssistant
from audiostream import AudioStreamer


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









class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech to Text Converter")
        self.setMinimumSize(1200, 600)  # Increased width for 3 columns
        
        # Initialize managers
        self.db_manager = DatabaseManager()
        self.translation_manager = TranslationManager()
        self.ai_assistant = AIAssistant()  # Initialize AI Assistant
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create control panel
        control_panel = QHBoxLayout()
        
        # Left side controls
        left_controls = QHBoxLayout()
        
        # Create language selection
        self.language_combo = QComboBox()
        self.language_combo.addItems(["English", "Chinese"])
        left_controls.addWidget(QLabel("Language:"))
        left_controls.addWidget(self.language_combo)
        self.language_combo.currentTextChanged.connect(self.language_changed)
        
        # Create model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(WHISPER_MODELS.keys())
        left_controls.addWidget(QLabel("Model:"))
        left_controls.addWidget(self.model_combo)
        self.model_combo.currentTextChanged.connect(self.model_changed)
        
        # Right side controls
        right_controls = QHBoxLayout()
        
        # Create role selection
        self.role_combo = QComboBox()
        self.role_combo.addItems([
            "General Assistant",
            "Technical Interviewer",
            "HR Interviewer",
            "Meeting Participant",
            "Student",
            "Teacher",
            "Custom..."
        ])
        self.role_combo.currentTextChanged.connect(self.role_changed)
        right_controls.addWidget(QLabel("AI Role:"))
        right_controls.addWidget(self.role_combo)
        
        # Create translation and AI answer checkboxes
        self.translate_check = QCheckBox("Auto Translate")
        self.answer_check = QCheckBox("AI Answer Questions")
        self.api_config_button = QPushButton("API Settings")
        right_controls.addWidget(self.translate_check)
        right_controls.addWidget(self.answer_check)
        right_controls.addWidget(self.api_config_button)
        self.api_config_button.clicked.connect(self.configure_api)
        
        # Create start/stop button
        self.toggle_button = QPushButton("Start Streaming")
        right_controls.addWidget(self.toggle_button)
        
        # Add controls to main control panel
        control_panel.addLayout(left_controls)
        control_panel.addStretch()
        control_panel.addLayout(right_controls)
        
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
        
        # AI Answer section
        answer_section = QVBoxLayout()
        answer_section.addWidget(QLabel("AI Answers:"))
        self.answer_output = QTextEdit()
        self.answer_output.setReadOnly(True)
        answer_section.addWidget(self.answer_output)
        text_container.addLayout(answer_section)
        
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
        
        # Store conversation context
        self.conversation_context = []
        self.context_window_size = 100  # Store last 10 exchanges
        
        # Store current role description
        self.current_role = "You are a helpful assistant providing concise answers based on conversation context."
        
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
        
        # Connect new signals
        self.transcriber.word_ready.connect(self.handle_word)
        self.transcriber.sentence_complete.connect(self.handle_sentence_complete)
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
    
    def handle_word(self, word, timestamp, is_translation):
        # Get the appropriate text widget
        text_widget = self.translation_output if is_translation else self.original_output
        
        # Clean up the word
        word = word.strip()
        if not word:  # Skip empty words
            return
            
        # Check if it's a punctuation mark
        is_punctuation = word in ['.', '!', '?', '。', '！', '？', ',', '，', '、']
        
        # Get current text and determine if we need a new line
        current_text = text_widget.toPlainText()
        needs_new_line = current_text == '' or current_text.endswith('\n')
        
        # Start new line with timestamp if needed (but not for lone punctuation)
        if needs_new_line and not is_punctuation:
            text_widget.insertPlainText(f"[{timestamp}] {word}")
        else:
            # Add space before word unless it's a punctuation mark
            if not is_punctuation:
                text_widget.insertPlainText(f" {word}")
            # else:
            #     text_widget.insertPlainText(word)
        
        # Scroll to bottom
        text_widget.verticalScrollBar().setValue(
            text_widget.verticalScrollBar().maximum()
        )
    
    def handle_sentence_complete(self, is_translation):
        # Get the appropriate text widget
        text_widget = self.translation_output if is_translation else self.original_output
        
        # Get the current line before adding newline
        current_text = text_widget.toPlainText()
        current_line = current_text.split('\n')[-1] if current_text else ""
        
        # Only add newline if the current line has content
        if current_text and not current_text.endswith('\n'):
            text_widget.insertPlainText('\n')
        
        # Handle question detection and answering for original text only
        if not is_translation:
            # Extract the actual text without timestamp
            if current_line.startswith('[') and ']' in current_line:
                actual_text = current_line[current_line.index(']')+1:].strip()
                
                # Add to conversation context
                self.conversation_context.append(actual_text)
                if len(self.conversation_context) > self.context_window_size * 2:
                    self.conversation_context.pop(0)
                
                # Check if it's a question and AI answering is enabled
                if self.answer_check.isChecked() and self.ai_assistant.is_question(actual_text):
                    timestamp = current_line[1:current_line.index(']')]
                    answer = self.ai_assistant.answer_question(
                        actual_text, 
                        self.get_context(),
                        self.current_role,
                        timestamp,
                        self.language_combo.currentText()
                    )
                    self.answer_output.append(answer)
                    self.answer_output.verticalScrollBar().setValue(
                        self.answer_output.verticalScrollBar().maximum()
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
                    # Configure both translation manager and AI assistant
                    self.translation_manager.configure(base_url, api_key, model_name)
                    self.ai_assistant.configure(base_url, api_key, model_name)
                    self.status_label.setText(f"API configured successfully (Model: {model_name})")
                except Exception as e:
                    self.status_label.setText(f"API configuration failed: {str(e)}")
    
    def get_context(self):
        # Get the last few exchanges as context
        context = []
        for text in self.conversation_context[-self.context_window_size:]:
            context.append(text)
        return "\n".join(context)
    
    def role_changed(self, new_role):
        if new_role == "Custom...":
            dialog = QDialog(self)
            dialog.setWindowTitle("Custom Role Configuration")
            layout = QVBoxLayout(dialog)
            
            # Add role description input
            layout.addWidget(QLabel("Enter custom role description:"))
            role_input = QTextEdit()
            role_input.setPlaceholderText("Describe the role and behavior of the AI assistant...")
            role_input.setMinimumHeight(100)
            layout.addWidget(role_input)
            
            # Add buttons
            button_box = QHBoxLayout()
            save_button = QPushButton("Save")
            cancel_button = QPushButton("Cancel")
            button_box.addWidget(save_button)
            button_box.addWidget(cancel_button)
            layout.addLayout(button_box)
            
            # Connect buttons
            save_button.clicked.connect(dialog.accept)
            cancel_button.clicked.connect(dialog.reject)
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                custom_role = role_input.toPlainText().strip()
                if custom_role:
                    self.current_role = custom_role
                else:
                    # Reset to default if empty
                    self.role_combo.setCurrentText("General Assistant")
            else:
                # Reset to default if cancelled
                self.role_combo.setCurrentText("General Assistant")
        else:
            # Predefined roles
            roles = {
                "General Assistant": "You are a helpful assistant providing concise answers based on conversation context.",
                "Technical Interviewer": "You are a technical interviewer assessing candidate's technical skills. Focus on technical accuracy and depth of knowledge in answers.",
                "HR Interviewer": "You are an HR interviewer evaluating candidate fit. Focus on soft skills, experience, and behavioral aspects in answers.",
                "Meeting Participant": "You are a meeting participant helping to clarify points and summarize discussions.",
                "Student": "You are a student assistant helping to understand and explain concepts in an educational context.",
                "Teacher": "You are a teacher providing clear explanations and educational context in your answers."
            }
            self.current_role = roles.get(new_role, roles["General Assistant"])

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 