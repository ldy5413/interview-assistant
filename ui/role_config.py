from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QTextEdit,
    QHBoxLayout, QPushButton
)

ROLE_DESCRIPTIONS = {
    "General Assistant": "You are a helpful assistant providing concise answers based on conversation context.",
    "Technical Interviewer": "You are a technical interviewer assessing candidate's technical skills. Focus on technical accuracy and depth of knowledge in answers.",
    "HR Interviewer": "You are an HR interviewer evaluating candidate fit. Focus on soft skills, experience, and behavioral aspects in answers.",
    "Meeting Participant": "You are a meeting participant helping to clarify points and summarize discussions.",
    "Student": "You are a student assistant helping to understand and explain concepts in an educational context.",
    "Teacher": "You are a teacher providing clear explanations and educational context in your answers."
}

class RoleConfigDialog(QDialog):
    """Dialog for entering a custom AI role description."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Role Configuration")
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Enter custom role description:"))
        self.role_input = QTextEdit()
        self.role_input.setPlaceholderText(
            "Describe the role and behavior of the AI assistant..."
        )
        self.role_input.setMinimumHeight(100)
        layout.addWidget(self.role_input)

        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        cancel_button = QPushButton("Cancel")
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        save_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)

    def get_role_description(self) -> str:
        """Return the text entered by the user."""
        return self.role_input.toPlainText().strip()

