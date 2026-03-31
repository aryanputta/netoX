FROM python:3.11-slim

# Non-root user for security
RUN groupadd -r netobot && useradd -r -g netobot netobot

WORKDIR /app

# System deps for matplotlib (headless) and numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Headless matplotlib backend
ENV MPLBACKEND=Agg


COPY . .

RUN chown -R netobot:netobot /app
USER netobot

EXPOSE 5005/udp
EXPOSE 5006/udp

ENTRYPOINT ["python", "main.py"]
CMD ["--no-viz", "--controller", "pid"]
