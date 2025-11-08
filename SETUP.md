# Setup Guide

Complete setup instructions for the Drive RAG system using entirely open source components.

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Google Drive API credentials (Service Account) - only for document ingestion
- LLM provider (choose one):
  - **Ollama** (default) - Free, open-source CLI tool
  - **LM Studio** - User-friendly GUI for local models
  - **Gemini** - Google's cloud API (fast and affordable)
  - Any OpenAI-compatible API

**No paid API services required!**

## Installation

### 1. Install LLM Provider

#### Option A: Ollama (CLI - Default)

Download and install Ollama from [https://ollama.com](https://ollama.com)

```bash
# Pull a model (choose one)
ollama pull mistral      # Recommended: 7B params, good quality
ollama pull llama3.1     # Alternative: 8B params, excellent
ollama pull phi3         # Lightweight: 3.8B params

# Verify Ollama is running
ollama list
```

#### Option B: LM Studio (GUI - Beginner Friendly)

1. Download LM Studio from [https://lmstudio.ai](https://lmstudio.ai)
2. Install and launch LM Studio
3. Download a model from the Search tab (e.g., Mistral 7B, Llama 3.1)
4. Go to the "Local Server" tab
5. Click "Start Server" (default: http://localhost:1234)
6. Note the model name shown in LM Studio

#### Option C: Google Gemini (Cloud API)

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the API key
5. No installation needed - works immediately

**Pros**: Fast, affordable, no local compute needed  
**Cons**: Requires internet, costs money (but very cheap with free tier)  
**Best for**: Quick testing, production deployments without local GPU

### 2. Clone and Configure

```bash
cd drive-rag
cp .env.example .env
```

Edit `.env` with your configuration:

**For Ollama (default):**
```env
# Google Drive API (for document ingestion only)
GOOGLE_APPLICATION_CREDENTIALS=/secrets/sa.json
ROOT_FOLDER_ID=your_drive_folder_id

# Database
DB_URL=postgresql+psycopg://rag_user:rag_password@localhost:5432/rag_db

# Embedding Model (local, no API key needed)
EMBEDDING_MODEL=intfloat/multilingual-e5-large

# LLM Provider - Ollama
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral

# Reranker (local, no API key needed)
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

**For LM Studio:**
```env
# Google Drive API (for document ingestion only)
GOOGLE_APPLICATION_CREDENTIALS=/secrets/sa.json
ROOT_FOLDER_ID=your_drive_folder_id

# Database
DB_URL=postgresql+psycopg://rag_user:rag_password@localhost:5432/rag_db

# Embedding Model (local, no API key needed)
EMBEDDING_MODEL=intfloat/multilingual-e5-large

# LLM Provider - LM Studio
LLM_PROVIDER=openai
OPENAI_API_BASE=http://localhost:1234/v1
OPENAI_API_KEY=lm-studio
OPENAI_MODEL=local-model  # Use the model name shown in LM Studio

# Reranker (local, no API key needed)
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

**For Google Gemini:**
```env
# Google Drive API (for document ingestion only)
GOOGLE_APPLICATION_CREDENTIALS=/secrets/sa.json
ROOT_FOLDER_ID=your_drive_folder_id

# Database
DB_URL=postgresql+psycopg://rag_user:rag_password@localhost:5432/rag_db

# Embedding Model (local, no API key needed)
EMBEDDING_MODEL=intfloat/multilingual-e5-large

# LLM Provider - Gemini
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-api-key-from-ai-studio
GEMINI_MODEL=gemini-1.5-flash  # or gemini-1.5-pro for best quality

# Reranker (local, no API key needed)
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

**For other OpenAI-compatible APIs:**
```env
# LLM Provider - OpenAI-compatible
LLM_PROVIDER=openai
OPENAI_API_BASE=https://api.your-provider.com/v1
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=your-model-name
```

### 3. Google Drive Service Account Setup

**Note**: This is only needed for ingesting documents from Google Drive. The API itself uses no Google Cloud services.

#### A. Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Note your Project ID

#### B. Enable Drive API

1. Go to **APIs & Services** > **Library**
2. Search and enable: **Google Drive API**
3. Wait a few minutes for activation

#### C. Create Service Account

1. Go to **APIs & Services** > **Credentials**
2. Click **Create Credentials** > **Service Account**
3. Name: `drive-rag-service`
4. Description: `RAG system for Drive documents`
5. Click **Create and Continue**, then **Done**

#### D. Create and Download Key

1. Click on your new service account
2. Go to **Keys** tab
3. Click **Add Key** > **Create new key**
4. Select **JSON** format
5. Click **Create** (file downloads automatically)

#### E. Add Key to Project

```bash
# Create secrets directory
mkdir secrets

# Copy the downloaded JSON file
Copy-Item "C:\Users\YOUR_USER\Downloads\your-project-*.json" "secrets\sa.json"

# Verify
ls secrets/
```

#### F. Grant Drive Access

1. Open the downloaded JSON file
2. Find the `client_email` field
3. Copy this email address
4. In Google Drive:
   - Navigate to the folder you want to index
   - Right-click > **Share**
   - Paste the service account email
   - Set permission to **Viewer**
   - Uncheck "Notify people"
   - Click **Share**
5. Get the folder ID from the URL:
   ```
   https://drive.google.com/drive/folders/1abc...xyz
                                           ^^^^^^^^^ This is your folder ID
   ```

#### G. Verify Setup

```bash
python scripts/verify_drive_setup.py
```

### 4. Start Services

```bash
docker-compose up -d
```

This starts:
- PostgreSQL with pgvector (port 5432)
- Redis (port 6379)
- FastAPI server (port 8000)
- Celery worker

### 4. Verify Installation

```bash
curl http://localhost:8000/healthz
```

## Usage

### Document Ingestion

**Option 1: Selective Ingestion (Recommended)**

```bash
# Export file list
python scripts/list_drive_files.py --folder-id YOUR_FOLDER_ID --format csv > files.csv

# Edit files.csv to keep only desired files

# Ingest selected files
python scripts/ingest_from_csv.py --csv files.csv
```

**Option 2: API-based (Background Job)**

```bash
curl -X POST http://localhost:8000/ingest/start \
  -H "Content-Type: application/json" \
  -d '{"root_folder_id": "YOUR_FOLDER_ID"}'
```

### Querying

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the project timeline?",
    "multi_query": true,
    "top_k": 8
  }'
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/healthz` | GET | Health check |
| `/ingest/start` | POST | Start ingestion job |
| `/ingest/status/{job_id}` | GET | Check job status |
| `/ask` | POST | Question answering |
| `/search` | POST | Document search only |
| `/metrics` | GET | System metrics |

## Local Development

### Without Docker

```bash
# Install dependencies
pip install -r requirements.txt

# Start only database services
docker-compose up postgres redis -d

# Initialize database
psql -h localhost -U rag_user -d rag_db -f migrations/init.sql

# Start Celery worker
celery -A app.tasks.celery_app worker --loglevel=info

# Start API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Configuration

Key settings in `.env`:

### Chunking
### Chunking
- `MAX_CHUNK_TOKENS=400` - Target tokens per chunk
- `CHUNK_OVERLAP=60` - Overlap tokens between chunks

### Retrieval
- `TOPK_CANDIDATES=50` - Initial candidates from hybrid search
- `TOPK_CONTEXT=8` - Final chunks after reranking
- `ENABLE_MULTI_QUERY=true` - Generate query variations
- `ENABLE_HYDE=false` - Use hypothetical document embeddings

### Models (All Open Source)
- `EMBEDDING_MODEL=intfloat/multilingual-e5-large` - Local embedding model (1024 dim)
- `RERANKER_MODEL=BAAI/bge-reranker-v2-m3` - Local cross-encoder reranker

### LLM Configuration
- `LLM_PROVIDER=ollama` or `openai` - Choose your LLM provider
- **For Ollama**: `OLLAMA_MODEL=mistral`, `OLLAMA_BASE_URL=http://localhost:11434`
- **For LM Studio**: `OPENAI_API_BASE=http://localhost:1234/v1`, `OPENAI_MODEL=local-model`
- **For OpenAI/other**: `OPENAI_API_BASE=...`, `OPENAI_API_KEY=...`, `OPENAI_MODEL=...`

All models run locally with no API calls (except if using cloud OpenAI). First run will download models automatically:
- multilingual-e5-large: ~2.5GB
- bge-reranker-v2-m3: ~560MB
- LLM models (via Ollama/LM Studio): 3-8GB depending on model

## GPU Acceleration (Recommended)

For acceptable performance with local models:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Start database services only
docker-compose up postgres redis -d

# Run API locally for direct GPU access
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Performance with GPU**: 
- Embedding generation: ~50-100 docs/sec
- Reranking: 3-5x faster than CPU
- Overall query latency: <1.5s

**CPU-only performance**: Works but slow (~5-10 docs/sec for ingestion). GPU strongly recommended for production use.

## Troubleshooting

### Google Drive Access Issues

**"HttpError 404: File not found"**
- Share the Drive folder with service account email
- Find email in `secrets/sa.json` under `client_email`

**"Credentials file not found"**
- Verify `secrets/sa.json` exists
- Check path in `.env`

**"Permission denied"**
- Service account needs Viewer access
- Wait a few minutes after sharing

**Empty file list**
- Verify folder ID is correct
- Check files are not in Trash
- Run `python scripts/list_drive_files.py --folder-id YOUR_ID`

### Database Issues

Enable pgvector extension:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Check indexed documents:
```sql
SELECT COUNT(*) FROM documents;
SELECT COUNT(*) FROM chunks;
```

### Worker Issues

Check Redis connection:
```bash
redis-cli ping
```

View worker logs:
```bash
docker-compose logs celery_worker
```

## Evaluation

Run quality evaluation with Ragas:

```python
from app.eval.ragas_runner import run_evaluation_from_yaml

results = run_evaluation_from_yaml(
    yaml_path="tests/test_dataset.yaml",
    rag_system=your_rag_instance,
    output_path="reports/ragas_report.yaml"
)
```
