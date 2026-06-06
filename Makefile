# This Makefile is optional. The commands defined here can still be executed manually.

.PHONY: install test run lint clean

install:
	pip install -r requirements.txt

test:
	python -m unittest discover -s tests -v

run:
	streamlit run app.py

lint:
	python -m py_compile Agent.py app.py app_helpers.py theme.py

clean:
	rm -rf **pycache** */**pycache** .pytest_cache
