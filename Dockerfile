FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HOME=/tmp

WORKDIR /app

# system deps needed for build + runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       git \
    && rm -rf /var/lib/apt/lists/*

# install mzpricer from Git (Linux build!)
RUN pip install --upgrade pip \
    && pip install "mzpricer @ git+https://github.com/mzelenetz/mzpricer.git@master#subdirectory=mzpricer-py"

# copy your app
COPY pyproject.toml README.md ./
COPY src ./src

# install app deps
RUN pip install .

EXPOSE 8080

ENTRYPOINT []

CMD ["sh","-c","streamlit run /app/src/ui.py \
  --server.address=0.0.0.0 \
  --server.port=$PORT \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false"]
