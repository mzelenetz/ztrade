FROM node:20-slim AS web-build

WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build


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

# built frontend, served by FastAPI
COPY --from=web-build /web/dist ./web/dist

EXPOSE 8080

ENTRYPOINT []

CMD ["sh","-c","uvicorn src.api.main:app --host 0.0.0.0 --port $PORT"]
