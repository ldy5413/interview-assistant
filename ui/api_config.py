import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton
)

class APIConfigDialog(QDialog):
    """Dialog for configuring OpenAI API access."""
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

