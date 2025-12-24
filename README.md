# Ollama API Gateway# Ollama API Gateway# Ollama API Service



> **Secure, scalable, and monitored access to Ollama AI models with a modern React dashboard**



A production-ready API gateway for Ollama with FastAPI backend, PostgreSQL database, React frontend, and comprehensive monitoring. Features role-based access control, usage analytics, and an interactive playground for testing AI models.> **Secure, scalable, and monitored access to Ollama AI models with a modern React dashboard**A secure, production-ready API gateway for Ollama with authentication, authorization, rate limiting, PostgreSQL database, and comprehensive monitoring.



![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)

![License](https://img.shields.io/badge/license-MIT-green.svg)

![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)A production-ready API gateway for Ollama with FastAPI backend, PostgreSQL database, React frontend, and comprehensive monitoring. Features role-based access control, usage analytics, and an interactive playground for testing AI models.## Features



## 📸 Screenshots



- **Home Page**: API documentation with code examples![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)✅ **Authentication & Authorization**

- **Admin Dashboard**: Complete API key management and system analytics

- **User Dashboard**: Personal usage statistics and metrics![License](https://img.shields.io/badge/license-MIT-green.svg)- API key-based authentication

- **Playground**: Interactive AI model testing with customizable parameters

![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)- Role-based access control (Admin/User)

## ✨ Features

- Secure key generation and management

### Backend (FastAPI + PostgreSQL)

- 🔐 **API Key Authentication** - Secure Bearer token authentication## 📸 Screenshots

- 👥 **Role-Based Access Control** - Admin and user roles with different permissions

- 📊 **Usage Tracking** - Detailed statistics for requests, tokens, and response times✅ **PostgreSQL Database**

- ⚡ **Rate Limiting** - Configurable limits per API key

- 🗄️ **PostgreSQL Database** - Persistent storage with async support- **Home Page**: API documentation with code examples- Persistent storage for API keys and usage logs

- 📝 **Auto-Generated Docs** - Interactive Swagger UI and ReDoc

- 🔌 **Ollama Integration** - Direct integration with local Ollama instance- **Admin Dashboard**: Complete API key management and system analytics- Connection pooling for high performance



### Frontend (React + Vite)- **User Dashboard**: Personal usage statistics and metrics- Full ACID compliance and data integrity

- 🎨 **Modern UI** - Clean, responsive design with Tailwind CSS

- 🌓 **Dark Mode** - Full dark/light theme support- **Playground**: Interactive AI model testing with customizable parameters- Scalable to millions of requests

- 📈 **Real-time Charts** - Interactive usage analytics with Recharts

- 🎮 **Interactive Playground** - Test AI models with custom parameters

- 📱 **Mobile Responsive** - Works seamlessly on all devices

- 🚀 **Fast Performance** - Optimized with Vite build tool## ✨ Features✅ **Rate Limiting**



## 🏗️ Architecture- Configurable rate limits per API key



```### Backend (FastAPI + PostgreSQL)- Per-hour request tracking

ollama-api-gateway/

├── backend/                # FastAPI backend service- 🔐 **API Key Authentication** - Secure Bearer token authentication- Automatic rate limit enforcement

│   ├── main.py            # API endpoints and application logic

│   ├── database.py        # Database models and configuration- 👥 **Role-Based Access Control** - Admin and user roles with different permissions

│   ├── requirements.txt   # Python dependencies

│   ├── Dockerfile         # Backend container configuration- 📊 **Usage Tracking** - Detailed statistics for requests, tokens, and response times✅ **Monitoring & Analytics**

│   └── README.md          # Backend documentation

│- ⚡ **Rate Limiting** - Configurable limits per API key- Request tracking and logging

├── frontend/              # React frontend application

│   ├── src/- 🗄️ **PostgreSQL Database** - Persistent storage with async support- Usage statistics per API key

│   │   ├── components/   # Reusable React components

│   │   ├── pages/        # Page components- 📝 **Auto-Generated Docs** - Interactive Swagger UI and ReDoc- Response time monitoring

│   │   ├── context/      # React context providers

│   │   └── services/     # API service layer- 🔌 **Ollama Integration** - Direct integration with local Ollama instance- Model usage analytics

│   ├── public/assets/    # Static assets and logos

│   ├── Dockerfile        # Frontend container configuration- Historical data analysis

│   └── README.md         # Frontend documentation

│### Frontend (React + Vite)

├── docker-compose.yml    # Multi-container orchestration

├── .env                  # Environment variables (create from .env.example)- 🎨 **Modern UI** - Clean, responsive design with Tailwind CSS✅ **API Documentation**

├── .env.example          # Environment variables template

└── README.md            # This file- 🌓 **Dark Mode** - Full dark/light theme support- Interactive Swagger UI documentation

```

- 📈 **Real-time Charts** - Interactive usage analytics with Recharts- ReDoc alternative documentation

## 🚀 Quick Start with Docker (Recommended)

- 🎮 **Interactive Playground** - Test AI models with custom parameters- Complete API reference

### Prerequisites

- Docker and Docker Compose installed- 📱 **Mobile Responsive** - Works seamlessly on all devices

- Ollama running locally at http://localhost:11434

- Git (to clone the repository)- 🚀 **Fast Performance** - Optimized with Vite build tool## Quick Start



### 1. Clone and Setup



```bash## 🏗️ Architecture### Prerequisites

# Clone the repository

git clone https://github.com/yourusername/ollama-api-gateway.git

cd ollama-api-gateway

```- Python 3.11 or higher

# Copy and configure environment file

cp .env.example .envollama-api-gateway/- PostgreSQL 16 or higher

# Edit .env with your preferred editor if needed

```├── backend/                # FastAPI backend service- Docker (optional, for containerized deployment)



### 2. Start All Services│   ├── main.py            # API endpoints and application logic- Ollama running on accessible host



```bash│   ├── database.py        # Database models and configuration

# Build and start all containers

docker compose up -d│   ├── requirements.txt   # Python dependencies### 1. Installation



# View logs│   ├── Dockerfile         # Backend container configuration

docker compose logs -f

```│   └── README.md          # Backend documentation```bash



### 3. Access the Application│# Clone or navigate to the project directory



- **Frontend**: http://localhost:3000├── frontend/              # React frontend applicationcd ollama-api-service

- **Backend API**: http://localhost:8000

- **API Docs**: http://localhost:8000/docs│   ├── src/

- **PostgreSQL**: localhost:5432

│   │   ├── components/   # Reusable React components# Create virtual environment

### 4. Sign In

│   │   ├── pages/        # Page componentspython -m venv venv

Use the demo credentials:

- **Admin**: `demo-admin-key-12345`│   │   ├── context/      # React context providers

- **User**: `demo-user-key-67890`

│   │   └── services/     # API service layer# Activate virtual environment

## 📦 Services

│   ├── public/assets/    # Static assets and logos# On macOS/Linux:

The application consists of three Docker containers:

│   ├── Dockerfile        # Frontend container configurationsource venv/bin/activate

| Service | Technology | Port | Description |

|---------|------------|------|-------------|│   └── README.md         # Frontend documentation# On Windows:

| **postgres** | PostgreSQL 16 | 5432 | Database for API keys and logs |

| **ollama-api** | FastAPI + Python | 8000 | Backend API service |│# venv\Scripts\activate

| **frontend** | React + Vite | 3000 | User interface |

├── docker-compose.yml    # Multi-container orchestration

## ⚙️ Configuration

├── .env.example          # Environment variables template# Install dependencies

### Single Environment File

└── README.md            # This filepip install -r requirements.txt

The project uses **one centralized `.env` file** in the root directory for all services. This eliminates hardcoded values and provides a single source of truth.

``````

**Setup Steps:**



```bash

# Copy the example environment file## 🚀 Quick Start with Docker (Recommended)### 2. Database Setup

cp .env.example .env



# Edit with your configuration

nano .env  # or use your preferred editor### Prerequisites#### Option A: Using Docker Compose (Recommended)

```

- Docker and Docker Compose installed

### Environment Variables Reference

- Ollama running locally at http://localhost:11434```bash

The `.env` file contains all configuration for PostgreSQL, Backend, and Frontend:

- Git (to clone the repository)# Start PostgreSQL container

```env

# =============================================================================docker-compose up -d postgres

# PostgreSQL Database

# =============================================================================### 1. Clone and Setup

POSTGRES_DB=ollama_api              # Database name

POSTGRES_USER=postgres              # Database user# Wait for PostgreSQL to be ready (check logs)

POSTGRES_PASSWORD=postgres          # Database password (change in production!)

POSTGRES_PORT=5432                  # PostgreSQL port```bashdocker-compose logs -f postgres



# Database connection URL for backend# Clone the repository```

DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/ollama_api

git clone https://github.com/yourusername/ollama-api-gateway.git

# =============================================================================

# Backend APIcd ollama-api-gateway#### Option B: Local PostgreSQL Installation

# =============================================================================

SECRET_KEY=NKLdJTql1oGCsTOGqdpJGVQCkt6FntM5D5ffiODjqRc  # Generate new key!

ACCESS_TOKEN_EXPIRE_MINUTES=60      # Token expiration time

HOST=0.0.0.0                        # Server host# Copy environment file```bash

PORT=8000                           # Backend port

cp .env.example .env# macOS

# =============================================================================

# Ollama Configuration```brew install postgresql@16

# =============================================================================

# For Docker on macOS/Windows: use host.docker.internalbrew services start postgresql@16

# For Docker on Linux: use host.docker.internal or 172.17.0.1

# For local development: use localhost### 2. Start All Servicescreatedb ollama_api

OLLAMA_BASE_URL=http://host.docker.internal:11434



# =============================================================================

# Demo API Keys (Testing Only - Remove in Production)```bash# Ubuntu/Debian

# =============================================================================

DEMO_ADMIN_KEY=demo-admin-key-12345# Build and start all containerssudo apt-get update

DEMO_USER_KEY=demo-user-key-67890

docker compose up -dsudo apt-get install postgresql-16

# =============================================================================

# Frontendsudo systemctl start postgresql

# =============================================================================

VITE_API_URL=http://localhost:8000  # Backend API URL# View logssudo -u postgres createdb ollama_api

FRONTEND_PORT=3000                   # Frontend port

docker compose logs -f

# =============================================================================

# Docker```# Verify installation

# =============================================================================

DOCKER_PLATFORM=linux/amd64         # For Apple Silicon compatibilitypsql -U postgres -d ollama_api -c "SELECT version();"

```

### 3. Access the Application```

### Configuration Benefits



✅ **Single Source of Truth** - All environment variables in one `.env` file  

✅ **No Hardcoded Values** - `docker-compose.yml` references all values from `.env`  - **Frontend**: http://localhost:3000### 3. Configuration

✅ **Git Safe** - `.env` is in `.gitignore` and won't be committed  

✅ **Easy Updates** - Change values in one place, all services pick them up  - **Backend API**: http://localhost:8000

✅ **Environment Specific** - Easy to maintain dev/staging/prod configurations

- **API Docs**: http://localhost:8000/docsCopy `.env.example` to `.env` and configure your settings:

### Generate Secure Keys

- **PostgreSQL**: localhost:5432

```bash

# Generate a secure SECRET_KEY for production```bash

openssl rand -hex 32

### 4. Sign Incp .env.example .env

# Generate secure API keys

python3 -c "import secrets; print(secrets.token_urlsafe(32))"```

```

Use the demo credentials:

### Docker Compose Integration

- **Admin**: `demo-admin-key-12345`Edit the `.env` file:

The `docker-compose.yml` automatically reads all variables from `.env`:

- **User**: `demo-user-key-67890`

```yaml

services:```env

  postgres:

    environment:## 📦 Services# Security - Change this to a secure random string

      - POSTGRES_DB=${POSTGRES_DB}

      - POSTGRES_USER=${POSTGRES_USER}SECRET_KEY=your-super-secret-key-change-this-in-production

      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}

    ports:The application consists of three Docker containers:

      - "${POSTGRES_PORT}:5432"

# Database Configuration

  ollama-api:

    platform: ${DOCKER_PLATFORM}| Service | Technology | Port | Description |DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/ollama_api

    environment:

      - SECRET_KEY=${SECRET_KEY}|---------|------------|------|-------------|

      - DATABASE_URL=${DATABASE_URL}

      - OLLAMA_BASE_URL=${OLLAMA_BASE_URL}| **postgres** | PostgreSQL 16 | 5432 | Database for API keys and logs |# Set your Ollama host (Linux VM IP or localhost)

      # ... all other variables from .env

```| **ollama-api** | FastAPI + Python | 8000 | Backend API service |OLLAMA_BASE_URL=http://your-linux-vm-ip:11434



**No hardcoded values in docker-compose.yml!** All configuration comes from `.env`.| **frontend** | React + Vite | 3000 | User interface |



## 🎯 Usage# Demo API Keys (for testing - generate secure keys for production)



### API Endpoints## ⚙️ ConfigurationDEMO_ADMIN_KEY=your-secure-admin-key



#### Public EndpointsDEMO_USER_KEY=your-secure-user-key

```bash

# Chat completion### Environment Variables

POST /api/chat

Authorization: Bearer YOUR_API_KEY# Server configuration



# List available modelsCreate a `.env` file in the project root:HOST=0.0.0.0

GET /api/models

Authorization: Bearer YOUR_API_KEYPORT=8000



# Get usage statistics```env```

GET /api/stats

Authorization: Bearer YOUR_API_KEY# Backend Configuration

```

SECRET_KEY=your-secret-key-here**Generate secure demo keys:**

#### Admin Endpoints

```bashOLLAMA_BASE_URL=http://host.docker.internal:11434```bash

# List all API keys

GET /api/keysDATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/ollama_apipython3 -c "import secrets; print('DEMO_ADMIN_KEY=ollama-' + secrets.token_urlsafe(32))"

Authorization: Bearer ADMIN_API_KEY

python3 -c "import secrets; print('DEMO_USER_KEY=ollama-' + secrets.token_urlsafe(32))"

# Create new API key

POST /api/keys# Demo API Keys (for testing)```

Authorization: Bearer ADMIN_API_KEY

DEMO_ADMIN_KEY=demo-admin-key-12345

# Delete API key

DELETE /api/keys/{key_id}DEMO_USER_KEY=demo-user-key-67890### 4. Run the Service

Authorization: Bearer ADMIN_API_KEY

```



### Code Examples# Token Configuration**Note:** Database tables are created automatically on first startup - no manual migration needed!



#### cURLACCESS_TOKEN_EXPIRE_MINUTES=60

```bash

curl -X POST http://localhost:8000/api/chat \#### Option A: Using Docker Compose (Full Stack)

  -H "Authorization: Bearer demo-user-key-67890" \

  -H "Content-Type: application/json" \# Frontend Configuration  

  -d '{

    "model": "functiongemma",VITE_API_URL=http://localhost:8000```bash

    "messages": [

      {"role": "user", "content": "Hello!"}```# Start all services (PostgreSQL + API)

    ],

    "temperature": 0.7docker-compose up -d

  }'

```### Docker Compose Configuration



#### Python# View logs to confirm database initialization

```python

import requestsThe `docker-compose.yml` file orchestrates all services:docker-compose logs -f ollama-api



response = requests.post(- **Network**: All services communicate via `ollama-network`

    "http://localhost:8000/api/chat",

    headers={"Authorization": "Bearer demo-user-key-67890"},- **Volumes**: PostgreSQL data persisted in `postgres_data` volume# You should see:

    json={

        "model": "functiongemma",- **Health Checks**: PostgreSQL health check ensures database is ready# "Database initialized"

        "messages": [

            {"role": "user", "content": "Hello!"}- **Platform**: `linux/amd64` for Apple Silicon compatibility# "Initialized X demo API keys from environment"

        ],

        "temperature": 0.7

    }

)## 🎯 Usage# Stop services

print(response.json())

```docker-compose down



#### JavaScript### API Endpoints```

```javascript

const response = await fetch('http://localhost:8000/api/chat', {

  method: 'POST',

  headers: {#### Public Endpoints#### Option B: Run Locally

    'Authorization': 'Bearer demo-user-key-67890',

    'Content-Type': 'application/json'```bash

  },

  body: JSON.stringify({# Chat completion```bash

    model: 'functiongemma',

    messages: [POST /api/chat# Development mode

      { role: 'user', content: 'Hello!' }

    ],Authorization: Bearer YOUR_API_KEYpython main.py

    temperature: 0.7

  })

});

# List available models# Or using uvicorn directly

const data = await response.json();

console.log(data);GET /api/modelsuvicorn main:app --reload --host 0.0.0.0 --port 8000

```

Authorization: Bearer YOUR_API_KEY```

## 🛠️ Development



### Backend Development

# Get usage statisticsThe service will:

```bash

cd backendGET /api/stats1. ✅ Connect to PostgreSQL



# Create .env from example (or use root .env)Authorization: Bearer YOUR_API_KEY2. ✅ Create tables automatically (if they don't exist)

cp .env.example .env

```3. ✅ Initialize demo API keys (if configured in .env)

# Install dependencies

pip install -r requirements.txt4. ✅ Start serving requests



# Run development server#### Admin Endpoints

uvicorn main:app --reload --host 0.0.0.0 --port 8000

``````bashThe service will start at: `http://localhost:8000`



See [backend/README.md](backend/README.md) for detailed backend documentation.# List all API keys



### Frontend DevelopmentGET /api/keys### 5. Verify Installation



```bashAuthorization: Bearer ADMIN_API_KEY

cd frontend

```bash

# Install dependencies

npm install# Create new API key# Check health



# Create .env.local for developmentPOST /api/keyscurl http://localhost:8000/health

echo "VITE_API_URL=http://localhost:8000" > .env.local

Authorization: Bearer ADMIN_API_KEY

# Run development server

npm run dev# List models (using demo key)

```

# Delete API keycurl -X GET "http://localhost:8000/api/models" \

See [frontend/README.md](frontend/README.md) for detailed frontend documentation.

DELETE /api/keys/{key_id}  -H "Authorization: Bearer demo-admin-key-12345"

## 🔧 Management Commands

Authorization: Bearer ADMIN_API_KEY```

### Docker Commands

```

```bash

# Start services## API Documentation

docker compose up -d

### Code Examples

# Stop services

docker compose downOnce the service is running, access the interactive documentation:



# Rebuild containers (after .env changes)#### cURL

docker compose up --build -d

```bash- **Swagger UI**: http://localhost:8000/docs

# View logs

docker compose logs -f [service_name]curl -X POST http://localhost:8000/api/chat \- **ReDoc**: http://localhost:8000/redoc



# Restart a specific service  -H "Authorization: Bearer demo-user-key-67890" \

docker compose restart [service_name]

  -H "Content-Type: application/json" \## Demo API Keys

# Remove all containers and volumes

docker compose down -v  -d '{

```

    "model": "functiongemma",The service supports demo API keys for testing. Set them in your `.env` file:

### Database Commands

    "messages": [

```bash

# Access PostgreSQL      {"role": "user", "content": "Hello!"}### Admin Key

docker exec -it ollama-postgres psql -U postgres -d ollama_api

    ],```

# Backup database

docker exec ollama-postgres pg_dump -U postgres ollama_api > backup.sql    "temperature": 0.7DEMO_ADMIN_KEY=ollama-<your-secure-token>



# Restore database  }'Role: admin

docker exec -i ollama-postgres psql -U postgres ollama_api < backup.sql

``````Rate Limit: 1000 requests/hour



### Environment ManagementPermissions: Full access (create keys, view all stats, use API)



```bash#### Python```

# Validate .env file exists

test -f .env && echo ".env exists" || echo ".env missing - copy from .env.example"```python



# Show current environment variablesimport requests### User Key

docker compose config

```

# Update environment (rebuild containers)

docker compose up --build -dresponse = requests.post(DEMO_USER_KEY=ollama-<your-secure-token>

```

    "http://localhost:8000/api/chat",Role: user  

## 📊 Monitoring

    headers={"Authorization": "Bearer demo-user-key-67890"},Rate Limit: 100 requests/hour

### Health Checks

    json={Permissions: Use API, view own stats only

```bash

# Backend health        "model": "functiongemma",```

curl http://localhost:8000/health

        "messages": [

# Database connection

docker exec ollama-postgres pg_isready -U postgres            {"role": "user", "content": "Hello!"}**Security Note**: Demo keys are stored in the database on first startup. Change them in production!



# Frontend (check if responding)        ],

curl http://localhost:3000

```        "temperature": 0.7## API Usage Examples



### Logs    }



```bash)### Authentication

# All services

docker compose logs -fprint(response.json())



# Backend only```All requests require an API key in the Authorization header:

docker compose logs -f ollama-api



# Frontend only

docker compose logs -f frontend#### JavaScript```bash



# PostgreSQL only```javascriptAuthorization: Bearer <your-api-key>

docker compose logs -f postgres

```const response = await fetch('http://localhost:8000/api/chat', {```



## 🚨 Troubleshooting  method: 'POST',



### Common Issues  headers: {### 1. List Available Models



#### 1. Port Already in Use    'Authorization': 'Bearer demo-user-key-67890',

```bash

# Find process using port    'Content-Type': 'application/json'```bash

lsof -i :8000  # or :3000 or :5432

  },curl -X GET "http://localhost:8000/api/models" \

# Kill process

kill -9 <PID>  body: JSON.stringify({  -H "Authorization: Bearer <your-api-key>"

```

    model: 'functiongemma',```

#### 2. Docker exec format error (Apple Silicon)

Ensure `DOCKER_PLATFORM=linux/amd64` is set in `.env` file.    messages: [



#### 3. Ollama Connection Failed      { role: 'user', content: 'Hello!' }### 2. Generate Text

```bash

# Check Ollama is running    ],

curl http://localhost:11434/api/tags

    temperature: 0.7```bash

# Verify OLLAMA_BASE_URL in .env

# Should be: http://host.docker.internal:11434 (in Docker)  })curl -X POST "http://localhost:8000/api/generate" \

```

});  -H "Authorization: Bearer <your-api-key>" \

#### 4. Database Connection Issues

```bash  -H "Content-Type: application/json" \

# Check PostgreSQL logs

docker logs ollama-postgresconst data = await response.json();  -d '{



# Verify database existsconsole.log(data);    "model": "llama2",

docker exec ollama-postgres psql -U postgres -l

```    "prompt": "Explain what is machine learning in simple terms",

# Check DATABASE_URL in .env matches postgres credentials

```    "temperature": 0.7,



#### 5. Frontend Not Loading## 🛠️ Development    "stream": false

```bash

# Check frontend logs  }'

docker logs ollama-frontend

### Backend Development```

# Verify VITE_API_URL in .env is correct

# Should be: http://localhost:8000

```

```bash### 3. Chat Completion

#### 6. Environment Variables Not Working

```bashcd backend

# Verify .env file exists

ls -la .env```bash



# Show loaded environment# Install dependenciescurl -X POST "http://localhost:8000/api/chat" \

docker compose config

pip install -r requirements.txt  -H "Authorization: Bearer <your-api-key>" \

# Rebuild after .env changes

docker compose down  -H "Content-Type: application/json" \

docker compose up --build -d

```# Run development server  -d '{



## 🔐 Securityuvicorn main:app --reload --host 0.0.0.0 --port 8000    "model": "llama2",



### Production Recommendations```    "messages": [



1. **Generate Strong Secret Key**      {

```bash

openssl rand -hex 32See [backend/README.md](backend/README.md) for detailed backend documentation.        "role": "system",

```

Update `SECRET_KEY` in `.env`        "content": "You are a helpful assistant."



2. **Secure Database Password**### Frontend Development      },

```bash

# Generate strong password      {

openssl rand -base64 24

``````bash        "role": "user",

Update `POSTGRES_PASSWORD` in `.env` and `DATABASE_URL`

cd frontend        "content": "What is the capital of France?"

3. **Disable Demo Keys**

```env      }

# In .env, remove or comment out:

# DEMO_ADMIN_KEY=# Install dependencies    ],

# DEMO_USER_KEY=

```npm install    "temperature": 0.7,



4. **Use Environment-Specific .env Files**    "stream": false

- `.env.development`

- `.env.staging`# Run development server  }'

- `.env.production`

npm run dev```

5. **Enable HTTPS**

- Use reverse proxy (nginx, Caddy)```

- Configure SSL certificates

- Update `VITE_API_URL` to use `https://`### 4. Get Usage Statistics



6. **Secure .env File**See [frontend/README.md](frontend/README.md) for detailed frontend documentation.

```bash

# Set restrictive permissions```bash

chmod 600 .env

## 🔧 Management Commandscurl -X GET "http://localhost:8000/api/stats" \

# Ensure .env is in .gitignore

grep -q "^\.env$" .gitignore || echo ".env" >> .gitignore  -H "Authorization: Bearer <your-api-key>"

```

### Docker Commands```

## 📚 Documentation



- [Backend Documentation](backend/README.md) - FastAPI backend details

- [Frontend Documentation](frontend/README.md) - React frontend details```bash### 5. Create New API Key (Admin Only)

- [API Documentation](backend/API_DOCUMENTATION.md) - Complete API reference

- [Migration Guide](backend/MIGRATION_GUIDE.md) - Database migrations# Start services



## 🧪 Testingdocker compose up -d```bash



### Backend Testscurl -X POST "http://localhost:8000/api/keys" \

```bash

cd backend# Stop services  -H "Authorization: Bearer <your-admin-key>" \

pytest

```docker compose down  -H "Content-Type: application/json" \



### Frontend Tests  -d '{

```bash

cd frontend# Rebuild containers    "name": "Production API Key",

npm run test

```docker compose up --build -d    "role": "user",



### API Testing    "rate_limit": 500

Use the included Swagger UI at http://localhost:8000/docs

# View logs  }'

## 🚀 Deployment

docker compose logs -f [service_name]```

### Production Deployment



1. **Create Production .env**

```bash# Restart a specific service### 6. List All API Keys (Admin Only)

cp .env.example .env.production

docker compose restart [service_name]

# Edit .env.production with production values

nano .env.production```bash

```

# Remove all containers and volumescurl -X GET "http://localhost:8000/api/keys" \

2. **Generate Secure Keys**

```bashdocker compose down -v  -H "Authorization: Bearer <your-admin-key>"

# Generate SECRET_KEY

openssl rand -hex 32``````



# Generate secure database password

openssl rand -base64 24

### Database Commands### 7. Get All Usage Statistics (Admin Only)

# Generate secure API keys (or disable demo keys)

python3 -c "import secrets; print(secrets.token_urlsafe(32))"

```

```bash```bash

3. **Update Production Values**

```env# Access PostgreSQLcurl -X GET "http://localhost:8000/api/admin/stats" \

SECRET_KEY=<generated-secret-key>

POSTGRES_PASSWORD=<secure-password>docker exec -it ollama-postgres psql -U postgres -d ollama_api  -H "Authorization: Bearer <your-admin-key>"

DATABASE_URL=postgresql+asyncpg://postgres:<secure-password>@postgres:5432/ollama_api

VITE_API_URL=https://api.yourdomain.com```

OLLAMA_BASE_URL=https://ollama.yourdomain.com

DEMO_ADMIN_KEY=  # Empty to disable# Backup database

DEMO_USER_KEY=   # Empty to disable

```docker exec ollama-postgres pg_dump -U postgres ollama_api > backup.sql## Python Client Example



4. **Deploy with Production .env**

```bash

# Use production environment file# Restore database```python

docker compose --env-file .env.production up -d

```docker exec -i ollama-postgres psql -U postgres ollama_api < backup.sqlimport requests



### Scaling```import os



- **Horizontal Scaling**: Run multiple backend containers behind load balancer

- **Database**: Use connection pooling, read replicas

- **Caching**: Add Redis for session management## 📊 MonitoringAPI_BASE_URL = "http://localhost:8000"



## 🤝 ContributingAPI_KEY = os.getenv("DEMO_USER_KEY", "your-api-key-here")



1. Fork the repository### Health Checks

2. Create feature branch (`git checkout -b feature/amazing-feature`)

3. Commit changes (`git commit -m 'Add amazing feature'`)headers = {

4. Push to branch (`git push origin feature/amazing-feature`)

5. Open Pull Request```bash    "Authorization": f"Bearer {API_KEY}",



## 📝 License# Backend health    "Content-Type": "application/json"



This project is licensed under the MIT License - see the LICENSE file for details.curl http://localhost:8000/health}



## 👥 Authors



- **Your Name** - Initial work - [YourGitHub](https://github.com/yourusername)# Database connection# List models



## 🙏 Acknowledgmentsdocker exec ollama-postgres pg_isready -U postgresresponse = requests.get(f"{API_BASE_URL}/api/models", headers=headers)



- [Ollama](https://ollama.ai) - Local AI model runtimeprint("Available models:", response.json())

- [FastAPI](https://fastapi.tiangolo.com) - Modern Python web framework

- [React](https://react.dev) - UI framework# Frontend (check if responding)

- [Tailwind CSS](https://tailwindcss.com) - Utility-first CSS framework

- [Vite](https://vitejs.dev) - Next generation frontend toolingcurl http://localhost:3000# Generate text



## 📞 Support```generate_payload = {



- **Issues**: [GitHub Issues](https://github.com/yourusername/ollama-api-gateway/issues)    "model": "llama2",

- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ollama-api-gateway/discussions)

- **Email**: support@yourcompany.com### Logs    "prompt": "Write a haiku about programming",



---    "temperature": 0.8,



**Built with ❤️ using FastAPI, React, and Docker**```bash    "stream": False



**Version**: 1.0.0 | **Last Updated**: December 23, 2025# All services}


docker compose logs -f

response = requests.post(

# Backend only    f"{API_BASE_URL}/api/generate",

docker compose logs -f ollama-api    headers=headers,

    json=generate_payload

# Frontend only)

docker compose logs -f frontendprint("Generated text:", response.json())



# PostgreSQL only# Chat completion

docker compose logs -f postgreschat_payload = {

```    "model": "llama2",

    "messages": [

## 🚨 Troubleshooting        {"role": "user", "content": "Hello! How are you?"}

    ],

### Common Issues    "stream": False

}

#### 1. Port Already in Use

```bashresponse = requests.post(

# Find process using port    f"{API_BASE_URL}/api/chat",

lsof -i :8000  # or :3000 or :5432    headers=headers,

    json=chat_payload

# Kill process)

kill -9 <PID>print("Chat response:", response.json())

```

# Get usage stats

#### 2. Docker exec format error (Apple Silicon)response = requests.get(f"{API_BASE_URL}/api/stats", headers=headers)

Ensure `platform: linux/amd64` is set in docker-compose.ymlprint("Usage statistics:", response.json())

```

#### 3. Ollama Connection Failed

```bash## API Endpoints Reference

# Check Ollama is running

curl http://localhost:11434/api/tags### General Endpoints

- `GET /` - API information

# Verify OLLAMA_BASE_URL in .env- `GET /health` - Health check

# Should be: http://host.docker.internal:11434 (in Docker)- `GET /docs` - Swagger UI documentation

```- `GET /redoc` - ReDoc documentation



#### 4. Database Connection Issues### API Key Management (Admin only)

```bash- `POST /api/keys` - Create new API key

# Check PostgreSQL logs- `GET /api/keys` - List all API keys

docker logs ollama-postgres- `DELETE /api/keys/{key_preview}` - Revoke API key



# Verify database exists### Ollama Operations

docker exec ollama-postgres psql -U postgres -l- `GET /api/models` - List available models

```- `POST /api/generate` - Generate text

- `POST /api/chat` - Chat completion

#### 5. Frontend Not Loading

```bash### Monitoring

# Check frontend logs- `GET /api/stats` - Get your usage statistics

docker logs ollama-frontend- `GET /api/admin/stats` - Get all statistics (admin only)



# Verify VITE_API_URL is correct## Rate Limiting

# Should be: http://localhost:8000

```Each API key has a configurable rate limit (requests per hour):

- Default user rate limit: 100 requests/hour

## 🔐 Security- Default admin rate limit: 1000 requests/hour

- Custom limits can be set when creating API keys

### Production Recommendations

When rate limit is exceeded, the API returns:

1. **Generate Strong Secret Key**```json

```bash{

openssl rand -hex 32  "detail": "Rate limit exceeded"

```}

```

2. **Use Environment-Specific Keys**Status code: 429 (Too Many Requests)

- Never commit real API keys to version control

- Use different keys for dev/staging/prod## Security Best Practices



3. **Enable HTTPS**1. **Change the SECRET_KEY**: Always use a strong, randomly generated secret key in production

- Use reverse proxy (nginx, Caddy)2. **Secure Database**: Use strong passwords and restrict database access

- Configure SSL certificates3. **Use HTTPS**: Deploy behind a reverse proxy with SSL/TLS

4. **Rotate API Keys**: Regularly rotate API keys and revoke unused ones

4. **Database Security**5. **Monitor Usage**: Review usage statistics for unusual patterns

- Change default PostgreSQL password6. **Firewall**: Restrict access to your Ollama VM and PostgreSQL

- Restrict database access7. **Backup Database**: Regularly backup your PostgreSQL database

- Regular backups8. **Environment Variables**: Never commit `.env` file to version control



5. **Rate Limiting**## Database Management

- Configure appropriate rate limits

- Monitor for abuse### Backup Database



## 📚 Documentation```bash

# Using Docker

- [Backend Documentation](backend/README.md) - FastAPI backend detailsdocker exec ollama-postgres pg_dump -U postgres ollama_api > backup.sql

- [Frontend Documentation](frontend/README.md) - React frontend details

- [API Documentation](backend/API_DOCUMENTATION.md) - Complete API reference# Locally

- [Migration Guide](backend/MIGRATION_GUIDE.md) - Database migrationspg_dump -U postgres ollama_api > backup.sql

```

## 🧪 Testing

### Restore Database

### Backend Tests

```bash```bash

cd backend# Using Docker

pytestcat backup.sql | docker exec -i ollama-postgres psql -U postgres ollama_api

```

# Locally

### Frontend Testspsql -U postgres ollama_api < backup.sql

```bash```

cd frontend

npm run test### Access Database Console

```

```bash

### API Testing# Using Docker

Use the included Swagger UI at http://localhost:8000/docsdocker exec -it ollama-postgres psql -U postgres -d ollama_api



## 🚀 Deployment# Locally

psql -U postgres -d ollama_api

### Production Deployment```



1. **Update Environment Variables**### Useful Database Queries

```env

SECRET_KEY=<strong-random-key>```sql

OLLAMA_BASE_URL=<production-ollama-url>-- View all API keys

DATABASE_URL=<production-database-url>SELECT key, name, role, rate_limit, is_active, created_at FROM api_keys;

VITE_API_URL=<production-api-url>

```-- View usage statistics

SELECT api_key, COUNT(*) as total, AVG(response_time) as avg_time

2. **Build Production Images**FROM usage_logs GROUP BY api_key;

```bash

docker compose -f docker-compose.prod.yml build-- Recent requests

```SELECT * FROM usage_logs ORDER BY timestamp DESC LIMIT 20;



3. **Deploy with Docker Swarm / Kubernetes**-- Clean old logs (keep last 90 days)

- Use orchestration for high availabilityDELETE FROM usage_logs WHERE timestamp < NOW() - INTERVAL '90 days';

- Configure load balancing```

- Set up monitoring and logging

For more database information, see [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md).

### Scaling

## Production Deployment

- **Horizontal Scaling**: Run multiple backend containers behind load balancer

- **Database**: Use connection pooling, read replicas### Using Docker Compose (Recommended)

- **Caching**: Add Redis for session management

The included `docker-compose.yml` provides a complete stack with PostgreSQL:

## 🤝 Contributing

```bash

1. Fork the repository# Build and start all services

2. Create feature branch (`git checkout -b feature/amazing-feature`)docker-compose up -d

3. Commit changes (`git commit -m 'Add amazing feature'`)

4. Push to branch (`git push origin feature/amazing-feature`)# View logs

5. Open Pull Requestdocker-compose logs -f



## 📝 License# Stop services

docker-compose down

This project is licensed under the MIT License - see the LICENSE file for details.

# Stop and remove volumes (WARNING: deletes database)

## 👥 Authorsdocker-compose down -v

```

- **Your Name** - Initial work - [YourGitHub](https://github.com/yourusername)

### Using Docker Manually

## 🙏 Acknowledgments

Build the API service:

- [Ollama](https://ollama.ai) - Local AI model runtime

- [FastAPI](https://fastapi.tiangolo.com) - Modern Python web framework```bash

- [React](https://react.dev) - UI frameworkdocker build -t ollama-api-service .

- [Tailwind CSS](https://tailwindcss.com) - Utility-first CSS framework```

- [Vite](https://vitejs.dev) - Next generation frontend tooling

Run with external PostgreSQL:

## 📞 Support

```bash

- **Issues**: [GitHub Issues](https://github.com/yourusername/ollama-api-gateway/issues)docker run -d \

- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ollama-api-gateway/discussions)  --name ollama-api \

- **Email**: support@yourcompany.com  -p 8000:8000 \

  -e SECRET_KEY="your-secret-key" \

---  -e DATABASE_URL="postgresql+asyncpg://user:pass@host:5432/dbname" \

  -e OLLAMA_BASE_URL="http://your-ollama-host:11434" \

**Built with ❤️ using FastAPI, React, and Docker**  ollama-api-service

```

**Version**: 1.0.0 | **Last Updated**: December 23, 2025

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
