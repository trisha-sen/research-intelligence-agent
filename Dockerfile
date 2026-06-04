FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/abstracts.csv data/abstracts.csv
COPY models/ models/
COPY main.py .
COPY load_data.py .
COPY agent_tools.py .
COPY agent_tools_lg.py .
COPY agent_graph.py .
COPY embed_corpus.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
