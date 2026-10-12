FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements correctly from apps/backend
COPY apps/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy workspace
COPY . .

ENV PYTHONPATH="/app/apps/backend"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "apps.backend.src.api.api_v2_production:app", "--host", "0.0.0.0", "--port", "8000"]
