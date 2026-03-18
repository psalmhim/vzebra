from PyQt6.QtWidgets import QApplication, QVBoxLayout, QTextEdit, QWidget, QLabel, QPushButton
from PyQt6.QtCore import Qt
import sys

class InteractiveTerminal(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = []  # To store command history
        self.locals = {}  # Dictionary to maintain the local variables
        self.insertPlainText(">>> ")  # Prompt

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Return:
            # Get the current line (user input)
            input_text = self.toPlainText().splitlines()[-1].replace(">>> ", "")
            self.history.append(input_text)  # Store the command in history
            self.execute_command(input_text)
            self.insertPlainText("\n>>> ")  # New prompt for the next command
        else:
            super().keyPressEvent(event)  # Default behavior for other keys

    def execute_command(self, command):
        """
        Execute the user's command and show the result or error.
        """
        try:
            # Try to evaluate the command (e.g., variable name or expression)
            result = eval(command, globals(), self.locals)
            if result is not None:
                self.insertPlainText(str(result))
        except Exception as e:
            # If eval fails, try executing the command (e.g., assignments or statements)
            try:
                exec(command, globals(), self.locals)
            except Exception as exec_e:
                self.insertPlainText(f"\nError: {exec_e}")
        self.ensureCursorVisible()

