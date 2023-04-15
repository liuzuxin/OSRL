SHELL=/bin/bash
PROJECT_NAME=osrl
PROJECT_PATH=${PROJECT_NAME}/
PYTHON_FILES = $(shell find setup.py ${PROJECT_NAME} examples -type f -name "*.py")

check_install = python3 -c "import $(1)" || pip3 install $(1) --upgrade
check_install_extra = python3 -c "import $(1)" || pip3 install $(2) --upgrade

pytest:
	$(call check_install, pytest)
	$(call check_install, pytest_cov)
	$(call check_install, pytest_xdist)
	pytest test --cov ${PROJECT_PATH} --durations 0 -v --cov-report term-missing --color=yes

mypy:
	$(call check_install, mypy)
	mypy ${PROJECT_NAME}

lint:
	$(call check_install, flake8)
	$(call check_install_extra, bugbear, flake8_bugbear)
	flake8 ${PYTHON_FILES} --count --show-source --statistics --max-line-length=89

format:
	$(call check_install, isort)
	isort --line-width=89 ${PYTHON_FILES}
	$(call check_install, yapf)
	yapf --style="{column_limit: 89}" -ir  ${PYTHON_FILES}

check-codestyle:
	$(call check_install, isort)
	$(call check_install, yapf)
	isort --line-width=89 --check ${PYTHON_FILES} && yapf --style="{based_on_style: pep8, column_limit: 89}" -r -d ${PYTHON_FILES}


# commit-checks: check-codestyle

.PHONY: format check-codestyle # commit-checks