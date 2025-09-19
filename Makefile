.PHONY: install run

ifeq ($(OS),Windows_NT)
PYTHON_VENV = ./venv/Scripts/python
else
PYTHON_VENV = ./venv/bin/python
endif

install:
	@echo "Creating virtual environment..."
	python -m venv venv
	@echo "Choose device: cpu or gpu"
	@read -p "Enter device (cpu/gpu): " DEVICE; \
	if [ "$$DEVICE" = "gpu" ]; then \
		echo "Install ocr via NVIDIA GPU"; \
		if [ "$(OS)" = "Windows_NT" ]; then \
			echo "Running on Windows"; \
			$(PYTHON_VENV) -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/; \
		else \
			echo "Running on Unix-like OS"; \
			$(PYTHON_VENV) -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/; \
		fi; \
	else \
		echo "Using ocr via CPU"; \
		$(PYTHON_VENV) -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/; \
	fi
	$(PYTHON_VENV) -m pip install -r requirements.txt \

run:
	$(PYTHON_VENV) -m fastapi run src/main.py
