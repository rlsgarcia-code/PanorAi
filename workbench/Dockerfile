FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv/panorai

COPY pyproject.toml setup.py MANIFEST.in README.md LICENSE ./
COPY panorai ./panorai
COPY contracts ./contracts
COPY docs/_static/tutorials ./docs/_static/tutorials
COPY workbench ./workbench

RUN python -m pip install --no-cache-dir . \
    && python -m pip install --no-cache-dir './workbench[ops]'

WORKDIR /srv/panorai/workbench
EXPOSE 8000

CMD ["uvicorn", "panorai_workbench.api:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
