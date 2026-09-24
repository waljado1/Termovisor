FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libimage-exiftool-perl \
        libgl1 \
        libglib2.0-0 \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["python", "main.py"]