FROM python:3.11.9-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV USV_MODE=server

EXPOSE 5000

ENTRYPOINT ["python", "docker_entrypoint.py"]
CMD ["server"]