# 🔧 Environment Configuration Guide

Complete documentation for all environment variables in the Ollama API Gateway.

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Database Configuration](#database-configuration)
3. [Backend API Configuration](#backend-api-configuration)
4. [Ollama Configuration](#ollama-configuration)
5. [Demo & Testing](#demo--testing)
6. [Frontend Configuration](#frontend-configuration)
7. [Docker Configuration](#docker-configuration)
8. [Security Best Practices](#security-best-practices)
9. [Environment-Specific Examples](#environment-specific-examples)

---

## Quick Reference

| Variable | Default | Required | Environment |
|----------|---------|----------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@postgres:5432/ollama_api` | ✅ | All |
| `SECRET_KEY` | (generated) | ✅ | All |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | ✅ | All |
| `VITE_API_URL` | `http://localhost:8000` | ✅ | Frontend |
| `DEMO_ADMIN_KEY` | `demo-admin-key-12345` | ❌ | Dev/Test |
| `DEMO_USER_KEY` | `demo-user-key-67890` | ❌ | Dev/Test |

---

## Database Configuration

### `DATABASE_URL`

Complete connection string for PostgreSQL database.

**Format**: `postgresql+asyncpg://[USER]:[PASSWORD]@[HOST]:[PORT]/[DATABASE]`

**Docker (Default)**:
```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/ollama_api
```
- Uses internal Docker DNS to connect to `postgres` service
- Credentials match `POSTGRES_USER` and `POSTGRES_PASSWORD`

**Local Development**:
```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/ollama_api
```
- Connects to local PostgreSQL instance
- PostgreSQL must be running: `brew services start postgresql`

**Remote Database (AWS RDS)**:
```env
DATABASE_URL=postgresql+asyncpg://admin:securepassword@mydb.xxxxx.rds.amazonaws.com:5432/ollama_api
```
- Use fully qualified RDS endpoint
- Ensure security groups allow connection

**Remote Database (Google Cloud SQL)**:
```env
DATABASE_URL=postgresql+asyncpg://postgres:password@cloudsql-proxy:5432/ollama_api
```
- Use Cloud SQL Proxy
- Configure firewall rules in GCP Console

### Related Variables

```env
POSTGRES_DB=ollama_api
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_PORT=5432
```

**Usage**: Used by Docker Compose to initialize PostgreSQL container
**Only needed**: When running with Docker Compose

**Change these for production**:
```env
POSTGRES_USER=secure_username
POSTGRES_PASSWORD=$(openssl rand -base64 24)
```

---

## Backend API Configuration

### `SECRET_KEY`

**Purpose**: Cryptographic key for signing JWT tokens and securing sessions

**Default**: `NKLdJTql1oGCsTOGqdpJGVQCkt6FntM5D5ffiODjqRc`

**⚠️ IMPORTANT**: Change this value in production!

**Generate Secure Key**:
```bash
# Option 1: OpenSSL (recommended)
openssl rand -hex 32

# Option 2: Python
python3 -c "import secrets; print(secrets.token_hex(32))"

# Option 3: Online generator
# https://randomkeygen.com/ (select 256-bit hex)
```

**Example**:
```env
SECRET_KEY=a7c9f3b2e1d4f6a8c5b9e2d4f7a0b3c5d8e1f4a7b0c3d6e9f2a5b8c1d4e7f0
```

**Security Notes**:
- Never commit to version control
- Use different keys for different environments
- Rotate periodically (invalidates existing tokens)
- Store in secure secrets manager (AWS Secrets Manager, HashiCorp Vault)

---

### `ACCESS_TOKEN_EXPIRE_MINUTES`

**Purpose**: JWT token lifetime in minutes

**Default**: `60`

**Recommended Values**:
- Development: `60` or higher for convenience
- Production: `15` - `30` for security
- Very Secure: `5` - `10`

**Examples**:
```env
# Development (1 hour)
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Standard Production (30 minutes)
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Highly Secure (10 minutes)
ACCESS_TOKEN_EXPIRE_MINUTES=10
```

**Behavior**:
- Shorter = More secure, more frequent logins
- Longer = Better UX, less secure
- Users can refresh token before expiry

---

### `HOST`

**Purpose**: Network interface to bind the server to

**Default**: `0.0.0.0`

**Options**:

| Value | Behavior | Use Case |
|-------|----------|----------|
| `0.0.0.0` | Listen on all interfaces | Docker/Production |
| `127.0.0.1` | Localhost only | Local development |
| `192.168.1.100` | Specific IP | Specific interface |

**Examples**:
```env
# Docker (all interfaces)
HOST=0.0.0.0

# Local only
HOST=127.0.0.1

# Specific interface
HOST=192.168.1.100
```

---

### `PORT`

**Purpose**: Backend service port

**Default**: `8000`

**Change if**:
- Port 8000 is already in use
- Running multiple instances
- Using non-standard ports

**Examples**:
```env
# Standard development
PORT=8000

# Alternative port
PORT=8001

# High port number
PORT=9000
```

**Check port availability**:
```bash
# macOS/Linux
lsof -i :8000

# Windows
netstat -ano | findstr :8000
```

---

## Ollama Configuration

### `OLLAMA_BASE_URL`

**Purpose**: URL to reach the Ollama service

**Critical for Success**: This is the #1 source of connectivity issues!

**Environment-Specific URLs**:

#### Docker Desktop (macOS/Windows)
```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
```
- Use `host.docker.internal` (Docker DNS magic hostname)
- Default Ollama port is 11434
- Works with Ollama running on host machine

#### Docker on Linux
```env
# Option A: Gateway IP
OLLAMA_BASE_URL=http://172.17.0.1:11434

# Option B: Host network mode (better)
# In docker-compose.yml, add: network_mode: host
```
- Linux containers use gateway IP to reach host
- Alternatively, configure host networking

#### Local Development (No Docker)
```env
OLLAMA_BASE_URL=http://localhost:11434
```
- Direct localhost connection
- Ollama must be running locally

#### Remote Ollama Server
```env
# IP-based
OLLAMA_BASE_URL=http://192.168.1.100:11434

# DNS-based
OLLAMA_BASE_URL=http://ollama.example.com:11434

# Custom port
OLLAMA_BASE_URL=http://ollama.example.com:9000
```

**Troubleshooting Connectivity**:

```bash
# Test connection
curl http://localhost:11434/api/tags

# Pull a model
ollama pull llama2

# Check Ollama status
ollama ps

# View logs
ollama logs
```

**Common Errors**:

| Error | Cause | Solution |
|-------|-------|----------|
| `Connection refused` | Ollama not running | Start Ollama: `ollama serve` |
| `Cannot reach host` | Wrong URL | Check `OLLAMA_BASE_URL` |
| `Network error` | Firewall blocked | Allow port 11434 |
| `Empty model list` | No models installed | `ollama pull llama2` |

---

## Demo & Testing

### `DEMO_ADMIN_KEY` and `DEMO_USER_KEY`

**Purpose**: Pre-generated API keys for testing and development

**Default Values**:
```env
DEMO_ADMIN_KEY=demo-admin-key-12345
DEMO_USER_KEY=demo-user-key-67890
```

**Features**:
- Auto-created on first startup
- Exist with specific roles (admin/user)
- Rate limits for demo keys
  - Admin: 1000 req/hr
  - User: 100 req/hr

**Using Demo Keys**:

```bash
# Call API with admin key
curl -H "Authorization: Bearer demo-admin-key-12345" \
  http://localhost:8000/api/models

# Sign in to frontend
1. Go to http://localhost:3000
2. Paste demo key as password
3. Click Continue
```

**Disable Demo Keys (Recommended for Production)**:

```env
DEMO_ADMIN_KEY=
DEMO_USER_KEY=
```

Leave empty to disable automatic demo key creation.

**Generate Secure Demo Keys**:

```bash
python3 << 'EOF'
import secrets

admin_key = f"admin-{secrets.token_urlsafe(32)}"
user_key = f"user-{secrets.token_urlsafe(32)}"

print(f"DEMO_ADMIN_KEY={admin_key}")
print(f"DEMO_USER_KEY={user_key}")
EOF
```

---

## Frontend Configuration

### `VITE_API_URL`

**Purpose**: Base URL for backend API calls from frontend

**Critical**: Must match your backend URL!

**Environment-Specific URLs**:

#### Local Development
```env
VITE_API_URL=http://localhost:8000
```
- Frontend connects directly to backend
- Both running on localhost

#### Docker Development
```env
VITE_API_URL=http://localhost:8000
```
- Frontend in container connects to backend
- Backend exposed on localhost:8000

#### Production without NGINX
```env
VITE_API_URL=https://api.yourdomain.com
```
- Direct connection to API server
- Must use HTTPS in production

#### Production with NGINX
```env
VITE_API_URL=https://yourdomain.com
```
- NGINX proxies `/api` to backend
- Frontend and API on same domain

**NGINX Proxy Example** (in nginx.conf):
```nginx
location /api {
    proxy_pass http://backend-service:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
}
```

**Testing Connectivity**:

```bash
# Check if API is reachable
curl $VITE_API_URL/health

# Check with authorization
curl -H "Authorization: Bearer YOUR_KEY" \
  $VITE_API_URL/api/models
```

### `FRONTEND_PORT`

**Purpose**: Port for frontend development server

**Default**: `3000`

**Change if**:
- Port 3000 is in use
- Running multiple instances

**Examples**:
```env
FRONTEND_PORT=3000
FRONTEND_PORT=3001
FRONTEND_PORT=8080
```

---

## Docker Configuration

### `DOCKER_PLATFORM`

**Purpose**: Specify architecture for Docker image building

**Default**: `linux/amd64`

**Options**:
- `linux/amd64` - Intel/AMD processors (x86-64)
- `linux/arm64` - ARM processors (Apple Silicon, etc.)

**Usage**:
```env
# Intel/AMD processor
DOCKER_PLATFORM=linux/amd64

# Apple Silicon (M1/M2/M3)
DOCKER_PLATFORM=linux/arm64
```

**Auto-Detection**:
```bash
# Check your processor architecture
uname -m

# Results:
# x86_64 -> Use linux/amd64
# arm64 -> Use linux/arm64
# aarch64 -> Use linux/arm64
```

---

## Security Best Practices

### 1. Production Checklist

```env
# ✅ Change all defaults
SECRET_KEY=<generated-secure-key>
POSTGRES_PASSWORD=<strong-password>
DEMO_ADMIN_KEY=
DEMO_USER_KEY=

# ✅ Use HTTPS
VITE_API_URL=https://yourdomain.com

# ✅ Secure database connection
DATABASE_URL=postgresql+asyncpg://user:secure_password@secure-host:5432/db

# ✅ Shorter token expiry
ACCESS_TOKEN_EXPIRE_MINUTES=15
```

### 2. Database Security

```bash
# Generate strong password
openssl rand -base64 32

# Store securely
# - AWS Secrets Manager
# - HashiCorp Vault
# - Azure Key Vault
# - Google Secret Manager
```

### 3. API Key Rotation

```bash
# Regularly rotate admin keys
1. Create new admin key
2. Update configuration
3. Update dependent services
4. Revoke old key
5. Test everything
```

### 4. Environment Variable Storage

**❌ DO NOT**:
- Commit `.env` to git
- Share sensitive keys
- Log environment variables
- Use weak passwords

**✅ DO**:
- Use `.gitignore` for `.env`
- Store in secure secrets manager
- Rotate keys regularly
- Use strong, generated passwords
- Audit access logs

**.gitignore**:
```
.env
.env.local
.env.*.local
```

---

## Environment-Specific Examples

### Development Environment

```env
# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/ollama_api

# Security (use defaults for dev)
SECRET_KEY=NKLdJTql1oGCsTOGqdpJGVQCkt6FntM5D5ffiODjqRc
ACCESS_TOKEN_EXPIRE_MINUTES=60

# API
HOST=0.0.0.0
PORT=8000

# Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Demo Keys (enabled for testing)
DEMO_ADMIN_KEY=demo-admin-key-12345
DEMO_USER_KEY=demo-user-key-67890

# Frontend
VITE_API_URL=http://localhost:8000
FRONTEND_PORT=3000

# Docker
DOCKER_PLATFORM=linux/amd64
```

### Staging Environment

```env
# Database (managed service)
DATABASE_URL=postgresql+asyncpg://admin:secure_password@staging-db.example.com:5432/ollama_api
POSTGRES_PASSWORD=secure_password

# Security (unique keys)
SECRET_KEY=$(openssl rand -hex 32)
ACCESS_TOKEN_EXPIRE_MINUTES=30

# API
HOST=0.0.0.0
PORT=8000

# Ollama
OLLAMA_BASE_URL=http://ollama-staging.internal:11434

# Demo Keys (disabled)
DEMO_ADMIN_KEY=
DEMO_USER_KEY=

# Frontend
VITE_API_URL=https://staging-api.example.com
FRONTEND_PORT=3000

# Docker
DOCKER_PLATFORM=linux/amd64
```

### Production Environment

```env
# Database (secured connection)
DATABASE_URL=postgresql+asyncpg://prod_user:secure_prod_password@prod-db.example.com:5432/ollama_api
POSTGRES_PASSWORD=secure_prod_password

# Security (strong settings)
SECRET_KEY=<use-secrets-manager>
ACCESS_TOKEN_EXPIRE_MINUTES=15

# API
HOST=0.0.0.0
PORT=8000

# Ollama
OLLAMA_BASE_URL=http://ollama-prod.internal:11434

# Demo Keys (disabled)
DEMO_ADMIN_KEY=
DEMO_USER_KEY=

# Frontend (HTTPS with domain)
VITE_API_URL=https://api.yourdomain.com
FRONTEND_PORT=3000

# Docker
DOCKER_PLATFORM=linux/amd64
```

---

## Validation & Testing

### Check Configuration

```bash
# Load environment and test
source .env

# Validate database connection
python3 << 'EOF'
from sqlalchemy import create_engine
try:
    engine = create_engine(os.getenv('DATABASE_URL'))
    engine.execute('SELECT 1')
    print("✅ Database connection OK")
except Exception as e:
    print(f"❌ Database error: {e}")
EOF

# Validate Ollama connection
curl -s $OLLAMA_BASE_URL/api/tags > /dev/null && echo "✅ Ollama OK" || echo "❌ Ollama failed"

# Validate API response
curl -H "Authorization: Bearer $DEMO_ADMIN_KEY" \
  http://localhost:8000/health && echo "✅ API OK" || echo "❌ API failed"
```

---

## Troubleshooting Configuration

### Database Connection Issues

**Problem**: `connection refused`

**Solutions**:
```bash
# Check PostgreSQL is running
brew services list | grep postgres

# Verify connection string
psql "postgresql://user:password@localhost:5432/db"

# Check if port is open
lsof -i :5432
```

### Ollama Connection Issues

**Problem**: `unreachable` or `timeout`

**Solutions**:
```bash
# Test curl
curl http://localhost:11434/api/tags

# Check Ollama running
ollama ps

# Check port
lsof -i :11434

# Verify environment variable
echo $OLLAMA_BASE_URL
```

### Frontend API Connection Issues

**Problem**: Frontend can't reach backend

**Solutions**:
```bash
# Verify VITE_API_URL
grep VITE_API_URL .env

# Test backend directly
curl http://localhost:8000/health

# Check browser console (F12)
# Look for CORS errors

# Verify frontend container network
docker network ls
docker network inspect ollama-api-service_ollama-network
```

---

## Summary

✅ **Before running**:
1. Copy `.env.example` to `.env`
2. Review all variables above
3. Update for your environment
4. Validate connections
5. Start application

📚 **References**:
- [README.md](./README.md) - Main documentation
- [backend/README.md](./backend/README.md) - Backend docs
- [frontend/README.md](./frontend/README.md) - Frontend docs

🆘 **Need help?**
- Check logs: `docker compose logs -f`
- Review errors in console output
- Search GitHub issues
- Ask on discussions

---

**Last Updated**: January 3, 2026
