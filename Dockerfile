# Recipe for the LLM Gateway image: a box with Python, every package, and our code.
# Ollama is NOT inside the box: it keeps running on the host computer (see docker-compose.yml).

# Start from a small official image that already has Python 3.9 (same version as the project).
FROM python:3.9-slim

# All our files live in /app inside the container.
WORKDIR /app

# Upgrade pip first: the old pip in this image fails on PyTorch's download server
# (it rejects "typing-extensions" vs "typing_extensions" as a name mismatch).
RUN pip install --no-cache-dir --upgrade pip

# Install CPU-only PyTorch first. The default PyTorch download includes GPU (CUDA)
# support, which adds several GB and can't be used here anyway.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the packages. Copying requirements.txt on its own first means
# Docker can reuse this slow step when only our code changes (layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding model while building, so the container doesn't have to
# download it every time it starts.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy our code (files listed in .dockerignore are left out).
COPY . .

# Inside a container, "localhost" means the container itself, not your computer.
# host.docker.internal is Docker's name for the computer running Docker, where Ollama is.
ENV OLLAMA_URL=http://host.docker.internal:11434/api/chat

# The gateway listens on port 8000. 0.0.0.0 = accept connections from outside the container.
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
