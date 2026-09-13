FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb xauth libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
    libasound2 libpango-1.0-0 libcairo2 procps curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /bin/bash /bin/sh

RUN groupadd -g 1000 solver && \
    useradd -u 1000 -g solver -m -s /bin/bash solver

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chown -R 1000:1000 /app

USER 1000:1000

EXPOSE 8877

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 CMD curl -f http://127.0.0.1:8877/health || exit 1

ENTRYPOINT ["xvfb-run", "-a", "--server-args=-screen 0 1920x1080x24", "python3", "server.py"]
