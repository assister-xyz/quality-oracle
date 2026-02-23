FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir "mcp[server]>=1.3.0"

COPY server.py .

EXPOSE 8010

CMD ["python", "server.py"]
