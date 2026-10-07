FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir . && civicite index --docs data/demo --out /app/.civicite/index
ENV CIVICITE_INDEX=/app/.civicite/index
EXPOSE 8000
# Optional LLM: docker run -e CIVICITE_LLM_BASE_URL=http://host.docker.internal:11434/v1 -e CIVICITE_LLM_MODEL=llama3.1:8b ...
CMD ["civicite", "serve", "--host", "0.0.0.0", "--port", "8000"]
