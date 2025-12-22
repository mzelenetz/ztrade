FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY options ./options
COPY README.md ./README.md

RUN pip install --no-cache-dir .

EXPOSE 8501

CMD ["streamlit", "run", "options/ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
