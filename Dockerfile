FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

RUN useradd --create-home --uid 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements-space.txt ./requirements-space.txt
USER user
RUN python -m pip install --user --upgrade pip && \
    python -m pip install --user -r requirements-space.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
