FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY main.py ./main.py
COPY config/ ./config/

CMD ["python", "main.py"]