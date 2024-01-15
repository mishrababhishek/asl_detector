VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip

.PHONY: setup

setup:
	@echo "Running script at `date +'%Y-%m-%d %H:%M:%S'`"
	@echo "==========================="

	# Check if Python 3.11.7 is installed
	@python --version 2>&1 | grep "3.11.7" > /dev/null || (echo "Python 3.11.7 is not installed. Please install it first."; exit 1)
	@echo "Python 3.11.7 is installed."

	# Create a virtual environment
	@echo "Creating virtual environment in $(VENV_DIR)."
	@python -m venv $(VENV_DIR)

	# Activate the virtual environment and install pip version 23.3.2
	@echo "Activating virtual environment."
	@$(PIP) install --upgrade pip==23.3.2

	# Install project requirements
	@echo "Installing project requirements."
	@$(PIP) install opencv-python==4.9.0.80
	@$(PIP) install cvzone==1.6.1
	@$(PIP) install mediapipe==0.10.9
	@$(PIP) install PyQt5==5.15.10
	@$(PIP) install pyttsx3==2.90
	@$(PIP) install transformers==4.36.2
	@$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

	# Deactivate the virtual environment
	@echo "Deactivating virtual environment."
	@deactivate || true
