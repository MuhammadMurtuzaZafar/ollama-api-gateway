# Ollama API Service

A secure, production-ready API gateway for Ollama with authentication, authorization, rate limiting, PostgreSQL database, and comprehensive monitoring.

## Features

✅ **Authentication & Authorization**
- API key-based authentication
- Role-based access control (Admin/User)
- Secure key generation and management

✅ **PostgreSQL Database**
- Persistent storage for API keys and usage logs
- Connection pooling for high performance
- Full ACID compliance and data integrity
- Scalable to millions of requests

✅ **Rate Limiting**
- Configurable rate limits per API key
- Per-hour request tracking
- Automatic rate limit enforcement

✅ **Monitoring & Analytics**
- Request tracking and logging
- Usage statistics per API key
- Response time monitoring
- Model usage analytics
- Historical data analysis

✅ **API Documentation**
- Interactive Swagger UI documentation
- ReDoc alternative documentation
- Complete API reference

## Quick Start

### Prerequisites

- Python 3.11 or higher
- PostgreSQL 16 or higher
- Docker (optional, for containerized deployment)
- Ollama running on accessible host

### 1. Installation

```bash
# Clone or navigate to the project directory
cd ollama-api-service

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Database Setup

#### Option A: Using Docker Compose (Recommended)

```bash
# Start PostgreSQL container
docker-compose up -d postgres

# Wait for PostgreSQL to be ready (check logs)
docker-compose logs -f postgres
```

#### Option B: Local PostgreSQL Installation

```bash
# macOS
brew install postgresql@16
brew services start postgresql@16
createdb ollama_api

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install postgresql-16
sudo systemctl start postgresql
sudo -u postgres createdb ollama_api

# Verify installation
psql -U postgres -d ollama_api -c "SELECT version();"
```

### 3. Configuration

Copy `.env.example` to `.env` and configure your settings:

```bash
cp .env.example .env
```

Edit the `.env` file:

```env
# Security - Change this to a secure random string
SECRET_KEY=your-super-secret-key-change-this-in-production

# Database Configuration
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/ollama_api

# Set your Ollama host (Linux VM IP or localhost)
OLLAMA_BASE_URL=http://your-linux-vm-ip:11434

# Demo API Keys (for testing - generate secure keys for production)
DEMO_ADMIN_KEY=your-secure-admin-key
DEMO_USER_KEY=your-secure-user-key

# Server configuration
HOST=0.0.0.0
PORT=8000
```

**Generate secure demo keys:**
```bash
python3 -c "import secrets; print('DEMO_ADMIN_KEY=ollama-' + secrets.token_urlsafe(32))"
python3 -c "import secrets; print('DEMO_USER_KEY=ollama-' + secrets.token_urlsafe(32))"
```

### 4. Run the Service

**Note:** Database tables are created automatically on first startup - no manual migration needed!

#### Option A: Using Docker Compose (Full Stack)

```bash
# Start all services (PostgreSQL + API)
docker-compose up -d

# View logs to confirm database initialization
docker-compose logs -f ollama-api

# You should see:
# "Database initialized"
# "Initialized X demo API keys from environment"

# Stop services
docker-compose down
```

#### Option B: Run Locally

```bash
# Development mode
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The service will:
1. ✅ Connect to PostgreSQL
2. ✅ Create tables automatically (if they don't exist)
3. ✅ Initialize demo API keys (if configured in .env)
4. ✅ Start serving requests

The service will start at: `http://localhost:8000`

### 5. Verify Installation

```bash
# Check health
curl http://localhost:8000/health

# List models (using demo key)
curl -X GET "http://localhost:8000/api/models" \
  -H "Authorization: Bearer demo-admin-key-12345"
```

## API Documentation

Once the service is running, access the interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Demo API Keys

The service supports demo API keys for testing. Set them in your `.env` file:

### Admin Key
```
DEMO_ADMIN_KEY=ollama-<your-secure-token>
Role: admin
Rate Limit: 1000 requests/hour
Permissions: Full access (create keys, view all stats, use API)
```

### User Key
```
DEMO_USER_KEY=ollama-<your-secure-token>
Role: user  
Rate Limit: 100 requests/hour
Permissions: Use API, view own stats only
```

**Security Note**: Demo keys are stored in the database on first startup. Change them in production!

## API Usage Examples

### Authentication

All requests require an API key in the Authorization header:

```bash
Authorization: Bearer <your-api-key>
```

### 1. List Available Models

```bash
curl -X GET "http://localhost:8000/api/models" \
  -H "Authorization: Bearer <your-api-key>"
```

### 2. Generate Text

```bash
curl -X POST "http://localhost:8000/api/generate" \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "prompt": "Explain what is machine learning in simple terms",
    "temperature": 0.7,
    "stream": false
  }'
```

### 3. Chat Completion

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ],
    "temperature": 0.7,
    "stream": false
  }'
```

### 4. Get Usage Statistics

```bash
curl -X GET "http://localhost:8000/api/stats" \
  -H "Authorization: Bearer <your-api-key>"
```

### 5. Create New API Key (Admin Only)

```bash
curl -X POST "http://localhost:8000/api/keys" \
  -H "Authorization: Bearer <your-admin-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Production API Key",
    "role": "user",
    "rate_limit": 500
  }'
```

### 6. List All API Keys (Admin Only)

```bash
curl -X GET "http://localhost:8000/api/keys" \
  -H "Authorization: Bearer <your-admin-key>"
```

### 7. Get All Usage Statistics (Admin Only)

```bash
curl -X GET "http://localhost:8000/api/admin/stats" \
  -H "Authorization: Bearer <your-admin-key>"
```

## Python Client Example

```python
import requests
import os

API_BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("DEMO_USER_KEY", "your-api-key-here")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# List models
response = requests.get(f"{API_BASE_URL}/api/models", headers=headers)
print("Available models:", response.json())

# Generate text
generate_payload = {
    "model": "llama2",
    "prompt": "Write a haiku about programming",
    "temperature": 0.8,
    "stream": False
}

response = requests.post(
    f"{API_BASE_URL}/api/generate",
    headers=headers,
    json=generate_payload
)
print("Generated text:", response.json())

# Chat completion
chat_payload = {
    "model": "llama2",
    "messages": [
        {"role": "user", "content": "Hello! How are you?"}
    ],
    "stream": False
}

response = requests.post(
    f"{API_BASE_URL}/api/chat",
    headers=headers,
    json=chat_payload
)
print("Chat response:", response.json())

# Get usage stats
response = requests.get(f"{API_BASE_URL}/api/stats", headers=headers)
print("Usage statistics:", response.json())
```

## API Endpoints Reference

### General Endpoints
- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - Swagger UI documentation
- `GET /redoc` - ReDoc documentation

### API Key Management (Admin only)
- `POST /api/keys` - Create new API key
- `GET /api/keys` - List all API keys
- `DELETE /api/keys/{key_preview}` - Revoke API key

### Ollama Operations
- `GET /api/models` - List available models
- `POST /api/generate` - Generate text
- `POST /api/chat` - Chat completion

### Monitoring
- `GET /api/stats` - Get your usage statistics
- `GET /api/admin/stats` - Get all statistics (admin only)

## Rate Limiting

Each API key has a configurable rate limit (requests per hour):
- Default user rate limit: 100 requests/hour
- Default admin rate limit: 1000 requests/hour
- Custom limits can be set when creating API keys

When rate limit is exceeded, the API returns:
```json
{
  "detail": "Rate limit exceeded"
}
```
Status code: 429 (Too Many Requests)

## Security Best Practices

1. **Change the SECRET_KEY**: Always use a strong, randomly generated secret key in production
2. **Secure Database**: Use strong passwords and restrict database access
3. **Use HTTPS**: Deploy behind a reverse proxy with SSL/TLS
4. **Rotate API Keys**: Regularly rotate API keys and revoke unused ones
5. **Monitor Usage**: Review usage statistics for unusual patterns
6. **Firewall**: Restrict access to your Ollama VM and PostgreSQL
7. **Backup Database**: Regularly backup your PostgreSQL database
8. **Environment Variables**: Never commit `.env` file to version control

## Database Management

### Backup Database

```bash
# Using Docker
docker exec ollama-postgres pg_dump -U postgres ollama_api > backup.sql

# Locally
pg_dump -U postgres ollama_api > backup.sql
```

### Restore Database

```bash
# Using Docker
cat backup.sql | docker exec -i ollama-postgres psql -U postgres ollama_api

# Locally
psql -U postgres ollama_api < backup.sql
```

### Access Database Console

```bash
# Using Docker
docker exec -it ollama-postgres psql -U postgres -d ollama_api

# Locally
psql -U postgres -d ollama_api
```

### Useful Database Queries

```sql
-- View all API keys
SELECT key, name, role, rate_limit, is_active, created_at FROM api_keys;

-- View usage statistics
SELECT api_key, COUNT(*) as total, AVG(response_time) as avg_time
FROM usage_logs GROUP BY api_key;

-- Recent requests
SELECT * FROM usage_logs ORDER BY timestamp DESC LIMIT 20;

-- Clean old logs (keep last 90 days)
DELETE FROM usage_logs WHERE timestamp < NOW() - INTERVAL '90 days';
```

For more database information, see [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md).

## Production Deployment

### Using Docker Compose (Recommended)

The included `docker-compose.yml` provides a complete stack with PostgreSQL:

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Stop and remove volumes (WARNING: deletes database)
docker-compose down -v
```

### Using Docker Manually

Build the API service:

```bash
docker build -t ollama-api-service .
```

Run with external PostgreSQL:

```bash
docker run -d \
  --name ollama-api \
  -p 8000:8000 \
  -e SECRET_KEY="your-secret-key" \
  -e DATABASE_URL="postgresql+asyncpg://user:pass@host:5432/dbname" \
  -e OLLAMA_BASE_URL="http://your-ollama-host:11434" \
  ollama-api-service
```

### Using systemd (Linux)

Create `/etc/systemd/system/ollama-api.service`:

```ini
[Unit]
Description=Ollama API Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/ollama-api-service
Environment="PATH=/opt/ollama-api-service/venv/bin"
ExecStart=/opt/ollama-api-service/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable ollama-api.service
sudo systemctl start ollama-api.service
```

## Monitoring and Logging

The service logs all activities including:
- API key authentication attempts
- Request processing
- Errors and exceptions
- Rate limit violations

Logs are written to stdout and can be redirected to a file or logging service.

## Troubleshooting

### Database Connection Issues

```bash
# Check if PostgreSQL is running
docker-compose ps postgres
# or
brew services list | grep postgresql

# Test connection
psql -U postgres -h localhost -p 5432 -d ollama_api

# Check logs
docker-compose logs postgres
```

### Migration Errors

```bash
# Reset database (WARNING: deletes all data)
docker-compose down -v
docker-compose up -d postgres

# Restart the API (tables will be recreated automatically)
docker-compose up -d ollama-api
# or
python main.py
```

### Can't connect to Ollama backend

1. Check if Ollama is running on your Linux VM:
   ```bash
   curl http://your-vm-ip:11434/api/tags
   ```

2. Ensure Ollama is listening on all interfaces:
   ```bash
   # On your Linux VM
   OLLAMA_HOST=0.0.0.0:11434 ollama serve
   ```

3. Check firewall rules on your Linux VM

### Rate limit errors

Check your current usage:
```bash
curl -X GET "http://localhost:8000/api/stats" \
  -H "Authorization: Bearer your-api-key"
```

### Authentication errors

1. Verify your API key exists in the database:
   ```sql
   SELECT * FROM api_keys WHERE key = 'your-key';
   ```

2. Check if key is active:
   ```sql
   SELECT is_active FROM api_keys WHERE key = 'your-key';
   ```

## Project Structure

```
ollama-api-service/
├── main.py              # FastAPI application
├── database.py          # Database models and configuration
├── requirements.txt     # Python dependencies
├── docker-compose.yml   # Docker Compose configuration
├── Dockerfile          # Docker image definition
├── .env                # Environment variables (not in git)
├── .env.example        # Example environment variables
├── README.md           # This file
├── CHANGELOG.md        # Migration summary and changes
├── MIGRATION_GUIDE.md  # Database migration documentation
└── API_DOCUMENTATION.md # Detailed API documentation
```

## Additional Resources

- **[API Documentation](API_DOCUMENTATION.md)** - Detailed API reference
- **[Migration Guide](MIGRATION_GUIDE.md)** - PostgreSQL setup and migration
- **[Swagger UI](http://localhost:8000/docs)** - Interactive API docs (when running)
- **[ReDoc](http://localhost:8000/redoc)** - Alternative API docs (when running)

## License

MIT License - feel free to use this in your projects!

## Support

For issues and questions, please check the API documentation at `/docs` or review the logs for error messages.
