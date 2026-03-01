FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY proto/ proto/
COPY app/ app/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 50051

CMD ["python", "-m", "app.server"]
