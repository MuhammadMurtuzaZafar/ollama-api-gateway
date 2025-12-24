# Ollama API Gateway - Backend

FastAPI-based backend service for managing and monitoring Ollama AI model access with PostgreSQL database.

## 🚀 Features

- **RESTful API** - FastAPI framework with automatic OpenAPI documentation
- **Authentication** - API key-based authentication with role-based access control (Admin/User)
- **Database** - PostgreSQL with SQLAlchemy ORM and async support
- **Rate Limiting** - Configurable rate limits per API key
- **Usage Tracking** - Detailed statistics for requests, tokens, and response times
- **Ollama Integration** - Direct integration with Ollama AI models
- **Demo Keys** - Pre-configured admin and user keys for testing

## 📋 Prerequisites

- Python 3.11+
- PostgreSQL 16
- Ollama running locally (http://localhost:11434)
- Docker and Docker Compose (for containerized deployment)

## 🛠️ Technology Stack

- **Framework**: FastAPI 0.104.1
- **Database**: PostgreSQL with asyncpg and SQLAlchemy 2.0
- **Authentication**: Custom API key-based system
- **HTTP Client**: httpx for async requests to Ollama
- **CORS**: Enabled for frontend integration
- **Migration**: Alembic for database migrations

## 📁 Project Structure

```
backend/
├── main.py                 # FastAPI application and endpoints
├── database.py            # Database models and configuration
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── .env.example          # Environment variables template
├── API_DOCUMENTATION.md  # Detailed API documentation
└── MIGRATION_GUIDE.md    # Migration instructions
```

## ⚙️ Environment Variables

Create a `.env` file in the backend directory:

```env
# Security
SECRET_KEY=your-secret-key-here

# Ollama Configuration
OLLAMA_BASE_URL=http://host.docker.internal:11434

# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/ollama_api

# Demo Keys (for testing)
DEMO_ADMIN_KEY=demo-admin-key-12345
DEMO_USER_KEY=demo-user-key-67890

# Token Expiration
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

## 🐳 Docker Deployment (Recommended)

### Using Docker Compose (from project root)

1. **Start all services**:
```bash
docker compose up -d
```

2. **View logs**:
```bash
docker compose logs -f ollama-api
```

3. **Stop services**:
```bash
docker compose down
```

4. **Rebuild after changes**:
```bash
docker compose up --build -d
```

### Environment Variables in Docker

The `docker-compose.yml` file automatically configures the backend with:
- PostgreSQL connection
- Ollama URL (host.docker.internal:11434)
- Demo API keys for testing

## 🔧 Local Development (Without Docker)

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Setup PostgreSQL Database

```bash
# Create database
createdb ollama_api

# Or using psql
psql -U postgres
CREATE DATABASE ollama_api;
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 4. Run Database Migrations

```bash
# Initialize database tables
python -c "from database import init_db; import asyncio; asyncio.run(init_db())"
```

### 5. Start Development Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **OpenAPI Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📡 API Endpoints

### Public Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Generate chat completions |
| GET | `/api/models` | List available Ollama models |
| GET | `/api/stats` | Get usage statistics for your API key |

### Admin-Only Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/keys` | List all API keys |
| POST | `/api/keys` | Create new API key |
| DELETE | `/api/keys/{key_id}` | Delete API key |

## 🔑 Authentication

All API requests require an `Authorization` header with Bearer token:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "functiongemma",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### Demo API Keys

For testing purposes:
- **Admin**: `demo-admin-key-12345`
- **User**: `demo-user-key-67890`

## 🗄️ Database Models

### APIKey Table
- `id`: UUID primary key
- `key`: Hashed API key (indexed)
- `name`: Key description
- `role`: 'admin' or 'user'
- `rate_limit`: Requests per hour
- `is_active`: Boolean
- `created_at`: Timestamp

### RequestLog Table
- `id`: UUID primary key
- `api_key_id`: Foreign key to APIKey
- `endpoint`: Request endpoint
- `method`: HTTP method
- `status_code`: Response status
- `tokens_used`: Number of tokens
- `response_time`: Response duration
- `created_at`: Timestamp

## 📊 Monitoring & Health

### Health Check
```bash
curl http://localhost:8000/health
```

### Database Health
```bash
curl http://localhost:8000/api/stats \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## 🔍 Logging

The application uses Python's built-in logging:
- **INFO**: General application flow
- **ERROR**: Error conditions
- **DEBUG**: Detailed diagnostic information (development)

View logs in Docker:
```bash
docker logs ollama-api-service -f
```

## 🧪 Testing

### Test Chat Endpoint
```python
import requests

response = requests.post(
    "http://localhost:8000/api/chat",
    headers={"Authorization": "Bearer demo-user-key-67890"},
    json={
        "model": "functiongemma",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }
)
print(response.json())
```

### Test with cURL
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Authorization: Bearer demo-user-key-67890" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "functiongemma",
    "messages": [
      {"role": "user", "content": "What is AI?"}
    ],
    "temperature": 0.7
  }'
```

## 🚨 Troubleshooting

### Database Connection Issues
```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Check database logs
docker logs ollama-postgres
```

### Ollama Connection Issues
```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# Inside Docker, use host.docker.internal
# Set OLLAMA_BASE_URL=http://host.docker.internal:11434
```

### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>
```

## 📚 Additional Documentation

- [API Documentation](./API_DOCUMENTATION.md) - Detailed API reference
- [Migration Guide](./MIGRATION_GUIDE.md) - Database migration instructions
- [Main README](../README.md) - Project overview

## 🤝 Development Workflow

1. Make changes to code
2. Test locally with `uvicorn main:app --reload`
3. Test with Docker: `docker compose up --build`
4. Commit changes
5. Push to repository

## 📦 Dependencies

Key Python packages (see `requirements.txt` for full list):
- `fastapi==0.104.1` - Web framework
- `uvicorn==0.24.0` - ASGI server
- `sqlalchemy==2.0.25` - ORM
- `asyncpg==0.29.0` - PostgreSQL driver
- `httpx==0.25.2` - HTTP client
- `python-dotenv==1.0.0` - Environment management

## 🔐 Security Best Practices

1. **Never commit `.env` files** - Use `.env.example` as template
2. **Rotate API keys regularly** - Especially in production
3. **Use strong SECRET_KEY** - Generate with `openssl rand -hex 32`
4. **Enable HTTPS in production** - Use reverse proxy (nginx, Caddy)
5. **Monitor rate limits** - Prevent abuse
6. **Regular backups** - Backup PostgreSQL database

## 📝 License

[Your License Here]

## 👥 Contributors

[Your Team/Contributors]

---

**Created**: December 23, 2025  
**Version**: 1.0.0
