# 🚀 Ollama API Gateway

> **Secure, scalable, and monitored access to Ollama AI models with a modern React dashboard**

A production-ready API gateway for Ollama with FastAPI backend, PostgreSQL database, React frontend, and comprehensive monitoring. Features role-based access control, usage analytics, rate limiting, and an interactive playground for testing AI models.

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Node](https://img.shields.io/badge/node-18+-green.svg)

---

## 📸 Screenshots

### Home Page
![Home Page](./assets/home.png)

### Admin Dashboard
![Dashboard](./assets/dashboard.png)

---

## ✨ Key Features

### 🔐 Security & Authentication
- **API Key Authentication** - Secure Bearer token authentication
- **Role-Based Access Control** - Admin and user roles with different permissions
- **Secure Key Generation** - Cryptographically secure key generation
- **Rate Limiting** - Configurable per-user hourly limits

### 📊 Monitoring & Analytics
- **Usage Tracking** - Detailed statistics for requests and response times
- **Admin Dashboard** - View all users' request patterns
- **User Analytics** - Personal usage statistics and metrics
- **Real-time Charts** - Interactive data visualization
- **Historical Data** - Track usage over 24h and 7d periods

### 🗄️ Database & Performance
- **PostgreSQL** - Persistent storage with async support
- **Connection Pooling** - High-performance database queries
- **ACID Compliance** - Data integrity and consistency
- **Scalable Design** - Handle millions of requests

### 🎨 User Interface
- **Modern React UI** - Clean, responsive design with Tailwind CSS
- **Dark Mode** - Full dark/light theme support
- **Mobile Responsive** - Works seamlessly on all devices
- **Interactive Playground** - Test AI models with custom parameters
- **Real-time Updates** - Live statistics and analytics

### 📝 API Documentation
- **Swagger UI** - Interactive API documentation at `/docs`
- **ReDoc** - Alternative API documentation view
- **Auto-Generated Specs** - OpenAPI 3.0 specification

---

## 🏗️ Architecture

```
ollama-api-gateway/
├── backend/                      # FastAPI backend service
│   ├── main.py                   # API endpoints and business logic
│   ├── database.py               # SQLAlchemy models and async connections
│   ├── requirements.txt          # Python dependencies
│   ├── Dockerfile                # Backend container config
│   └── README.md                 # Backend documentation
│
├── frontend/                     # React + Vite frontend application
│   ├── src/
│   │   ├── components/           # Reusable React components
│   │   │   ├── Navbar.jsx
│   │   │   ├── Charts.jsx
│   │   │   ├── StatsCard.jsx
│   │   │   └── APIKeyForm.jsx
│   │   ├── pages/                # Page components
│   │   │   ├── Home.jsx
│   │   │   ├── AdminDashboard.jsx
│   │   │   ├── UserDashboard.jsx
│   │   │   ├── Playground.jsx
│   │   │   └── ApiDocs.jsx
│   │   ├── context/              # React context providers
│   │   │   ├── AuthContext.jsx
│   │   │   └── ThemeContext.jsx
│   │   ├── services/             # API service layer
│   │   │   └── api.js
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── public/                   # Static files
│   │   ├── favicon.svg
│   │   └── assets/
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── Dockerfile
│   └── README.md
│
├── docker-compose.yml            # Multi-container orchestration
├── .env                          # Environment variables (gitignored)
├── .env.example                  # Environment template
├── ENV_CONFIGURATION.md          # Detailed environment guide
└── README.md                     # This file
```

---

## 🚀 Quick Start

### Prerequisites

- **Docker & Docker Compose** - For containerized deployment
- **Ollama** - Running locally at http://localhost:11434
- **Git** - To clone the repository
- **Port 3000, 8000, 5432** - Must be available

### Option 1: Docker (Recommended)

#### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ollama-api-gateway.git
cd ollama-api-gateway
```

#### 2. Configure Environment
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your settings (see Configuration section below)
nano .env
```

#### 3. Start Services
```bash
# Build and start all containers
docker compose up -d

# Check if all services are running
docker compose ps

# View logs
docker compose logs -f
```

#### 4. Access Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Database**: localhost:5432

### Option 2: Local Development

#### 1. Backend Setup
```bash
cd backend

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the backend
python main.py
```

#### 2. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

---

## ⚙️ Environment Configuration

### Overview
Environment variables are configured through the `.env` file. This file controls all aspects of the application including database connections, API settings, and feature toggles.

### File Structure
```
.env                 # Your local configuration (never commit)
.env.example        # Template with all available variables
ENV_CONFIGURATION.md # Detailed documentation for each variable
```

### Quick Setup

#### 1. Create `.env` File
```bash
cp .env.example .env
```

#### 2. Essential Variables
The minimum required configuration:

```env
# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/ollama_api

# API Keys (for demo - generate new ones for production)
DEMO_ADMIN_KEY=demo-admin-key-12345
DEMO_USER_KEY=demo-user-key-67890

# Frontend
VITE_API_URL=http://localhost:8000
```

#### 3. Apply Configuration
```bash
# Docker automatically reads .env
docker compose up -d

# Or manually export for local development
export $(cat .env | xargs)
```

---

## 📋 Complete Environment Variables

### Database Configuration

```env
# PostgreSQL Connection
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/ollama_api

# Connection Details
POSTGRES_DB=ollama_api
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_PORT=5432
```

**Usage**: Controls how the backend connects to PostgreSQL
**Default**: Built-in PostgreSQL container
**Production**: Use managed database service (AWS RDS, Google Cloud SQL, etc.)

---

### Backend API Configuration

```env
# Security
SECRET_KEY=NKLdJTql1oGCsTOGqdpJGVQCkt6FntM5D5ffiODjqRc

# Token Settings
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Server
HOST=0.0.0.0
PORT=8000
```

**SECRET_KEY**: Used for JWT token signing
- Generate new: `openssl rand -hex 32`
- Must be different in production

**ACCESS_TOKEN_EXPIRE_MINUTES**: JWT token lifetime
- Default: 60 minutes
- Adjust based on security requirements

**HOST**: Interface to bind to
- `0.0.0.0` = Listen on all interfaces (Docker)
- `127.0.0.1` = Localhost only (development)

**PORT**: Backend service port
- Default: 8000
- Change if port is already in use

---

### Ollama Configuration

```env
# Ollama Service URL
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

**For Different Environments**:

| Environment | URL | Notes |
|---|---|---|
| **Docker (macOS/Windows)** | `http://host.docker.internal:11434` | Special Docker DNS |
| **Docker (Linux)** | `http://172.17.0.1:11434` | Host gateway IP |
| **Local Development** | `http://localhost:11434` | Direct localhost |
| **Remote Server** | `http://your-host:11434` | Use actual IP/domain |

---

### Demo API Keys

```env
# Demo Admin Key (for testing)
DEMO_ADMIN_KEY=demo-admin-key-12345

# Demo User Key (for testing)
DEMO_USER_KEY=demo-user-key-67890
```

**⚠️ Security Warning**:
- Change these keys in production
- Leave empty to disable demo keys: `DEMO_ADMIN_KEY=`
- Generate secure keys: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`

**Usage**:
- Access API: `curl -H "Authorization: Bearer demo-admin-key-12345" http://localhost:8000/api/models`
- Login frontend: Use key as password on sign-in page

---

### Frontend Configuration

```env
# Backend API URL
VITE_API_URL=http://localhost:8000

# Frontend Port
FRONTEND_PORT=3000
```

**VITE_API_URL**:
- **Development**: `http://localhost:8000`
- **Docker**: `http://localhost:8000` or `http://api:8000`
- **Production**: `https://api.yourdomain.com`
- **NGINX**: `https://yourdomain.com/api` (with proper proxy config)

**FRONTEND_PORT**:
- Default: 3000
- Change if port conflicts

---

### Docker Configuration

```env
# Platform Architecture
DOCKER_PLATFORM=linux/amd64
```

**Options**:
- `linux/amd64` - Intel/AMD processors (default)
- `linux/arm64` - Apple Silicon (M1/M2/M3)

---

## 🔐 First-Time Setup (Step by Step)

### Step 1: Create Admin Account
```bash
# Option A: Use Demo Admin Key
curl -X POST http://localhost:8000/api/generate \
  -H "Authorization: Bearer demo-admin-key-12345" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama2","prompt":"hello"}'

# Option B: Create new admin key via API
curl -X POST http://localhost:8000/api/keys \
  -H "Authorization: Bearer demo-admin-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Admin Key",
    "role": "admin",
    "rate_limit": 1000
  }'
```

### Step 2: Login to Frontend
1. Go to http://localhost:3000
2. Click "Sign In"
3. Paste your admin API key
4. Click "Continue"

### Step 3: Create User API Keys
1. Go to Admin Dashboard
2. Click "Create API Key"
3. Fill in the form:
   - **Name**: User description
   - **Role**: "user" or "admin"
   - **Rate Limit**: Requests per hour
4. Copy and save the key securely

### Step 4: Test with Playground
1. Go to Playground tab
2. Select a model (e.g., "llama2")
3. Enter a prompt
4. Click "Generate"
5. View response and usage stats

---

## 🔑 API Key Management

### Creating API Keys

**Via Dashboard**:
1. Admin Dashboard → "Create API Key"
2. Fill form
3. Copy key immediately (shown only once)

**Via API**:
```bash
curl -X POST http://localhost:8000/api/keys \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Production API Key",
    "role": "user",
    "rate_limit": 100
  }'
```

### Revoking API Keys
```bash
curl -X DELETE http://localhost:8000/api/keys/KEY_PREFIX \
  -H "Authorization: Bearer YOUR_ADMIN_KEY"
```

### Viewing API Keys
```bash
curl http://localhost:8000/api/keys \
  -H "Authorization: Bearer YOUR_ADMIN_KEY"
```

---

## 📊 Admin Dashboard Features

### Overview Tab
- **Total Users**: Count of active API keys
- **Total Requests**: All-time request count
- **24h Requests**: Last 24 hours
- **7d Requests**: Last 7 days
- **Avg Response Time**: Average across all requests
- **Top Users Chart**: Bar chart of high-request users
- **Response Time Chart**: Users with slow response times
- **Rate Limit Usage**: Current hourly usage per user

### User Analytics Tab
- **User Statistics Table**: Detailed per-user metrics
- **Model Usage**: Which models each user accesses
- **Endpoint Usage**: Which endpoints are most used
- **Last Active**: When user last made a request
- **User Cards**: Quick view of top 6 users

### API Keys Tab
- **List All Keys**: View all active API keys
- **Show/Hide Key**: Reveal full key with confirmation
- **Copy Key**: One-click copy to clipboard
- **Revoke Key**: Deactivate a key

---

## 🐳 Docker Commands Reference

```bash
# Start all services
docker compose up -d

# Stop all services
docker compose down

# View running services
docker compose ps

# View logs
docker compose logs -f

# Rebuild containers
docker compose up -d --build

# Rebuild specific service
docker compose up -d --build backend

# Execute command in container
docker compose exec ollama-api-service python -c "import secrets; print(secrets.token_urlsafe(32))"

# Remove volumes (WARNING: deletes database)
docker compose down -v
```

---

## 🔍 Troubleshooting

### Port Already in Use
```bash
# Check which process is using the port
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use different port in .env
```

### Database Connection Failed
```bash
# Wait for PostgreSQL to be ready
docker compose logs postgres

# Reset database
docker compose down -v
docker compose up -d
```

### Ollama Not Found
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Update OLLAMA_BASE_URL in .env
# Restart containers
docker compose restart
```

### Frontend Can't Connect to API
```bash
# Check VITE_API_URL in .env
# Verify backend is running
curl http://localhost:8000/health

# Check browser console for CORS errors
# Restart frontend
docker compose restart frontend
```

---

## 📚 Documentation Files

- **[ENV_CONFIGURATION.md](./ENV_CONFIGURATION.md)** - Detailed environment variables guide
- **[backend/README.md](./backend/README.md)** - Backend API documentation
- **[frontend/README.md](./frontend/README.md)** - Frontend setup guide
- **[API Documentation](http://localhost:8000/docs)** - Interactive Swagger UI

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙋 Support

For issues and questions:
- GitHub Issues: [Report a bug](https://github.com/yourusername/ollama-api-gateway/issues)
- Discussions: [Ask a question](https://github.com/yourusername/ollama-api-gateway/discussions)
- Documentation: [Read the docs](./ENV_CONFIGURATION.md)

---

## 🎯 Roadmap

- [ ] Multi-model support with load balancing
- [ ] Advanced analytics and reporting
- [ ] Webhook notifications for usage alerts
- [ ] API versioning support
- [ ] Custom domain and SSL support
- [ ] Advanced caching mechanisms

---

**Made with ❤️ for the Ollama community**
