FROM python:3.10.12-slim

RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry


WORKDIR /app

# Install dependencies
COPY pyproject.toml poetry.lock /app/
RUN poetry config virtualenvs.create false && poetry install --only main --no-root

COPY . /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["tail", "-f", "/dev/null"]