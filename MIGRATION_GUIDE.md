# PostgreSQL Migration Guide

This guide explains how to migrate from in-memory storage to PostgreSQL.

## What Changed

- **Replaced in-memory storage** with PostgreSQL database
- **Added SQLAlchemy** for ORM and async database operations
- **Added database models** for API keys and usage logs
- **Updated all endpoints** to use database queries
- **Added PostgreSQL** to docker-compose.yml

## Database Schema

### Tables

#### `api_keys`
- `id` - Primary key
- `key` - API key (unique, indexed)
- `name` - Key description
- `role` - User role (admin/user)
- `rate_limit` - Requests per hour
- `created_at` - Creation timestamp
- `is_active` - Soft delete flag

#### `usage_logs`
- `id` - Primary key
- `api_key` - API key (indexed)
- `endpoint` - Endpoint called
- `model` - Model used
- `response_time` - Response time in seconds
- `timestamp` - Request timestamp (indexed)
- `request_data` - Additional JSON data

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Database

Update your `.env` file with the database URL:

```bash
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/ollama_api
```

### 3. Set Up PostgreSQL

#### Option A: Using Docker Compose (Recommended)

```bash
docker-compose up -d postgres
```

#### Option B: Local PostgreSQL

Install PostgreSQL and create the database:

```bash
# Install PostgreSQL (macOS)
brew install postgresql@16

# Start PostgreSQL
brew services start postgresql@16

# Create database
createdb ollama_api
```

### 4. Start the Application

**Tables are created automatically on first startup!**

```bash
# Using Docker Compose
docker-compose up -d

# Or run locally
python main.py
```

On first startup, you'll see:
```
Starting Ollama API Service...
Database initialized
Initialized X demo API keys from environment
```

The application will:
1. Connect to PostgreSQL
2. Create all tables (if they don't exist)
3. Initialize demo API keys (from .env)
4. Start serving requests

## Development

### Database Connection

The application uses async PostgreSQL connection with connection pooling:

```python
DATABASE_URL = "postgresql+asyncpg://user:password@host:port/database"
```

### Running Migrations

To reset the database:

```bash
# Stop the app
docker-compose down

# Remove volumes
docker-compose down -v

# Restart
docker-compose up -d
```

### Accessing PostgreSQL

```bash
# Using Docker
docker exec -it ollama-postgres psql -U postgres -d ollama_api

# Locally
psql -U postgres -d ollama_api
```

### Useful SQL Queries

```sql
-- View all API keys
SELECT key, name, role, rate_limit, is_active, created_at FROM api_keys;

-- View usage statistics
SELECT api_key, COUNT(*) as total_requests, AVG(response_time) as avg_time
FROM usage_logs
GROUP BY api_key;

-- View recent requests
SELECT * FROM usage_logs ORDER BY timestamp DESC LIMIT 10;

-- Clean up old logs (older than 30 days)
DELETE FROM usage_logs WHERE timestamp < NOW() - INTERVAL '30 days';
```

## Benefits of PostgreSQL

1. **Persistence** - Data survives restarts
2. **Scalability** - Handle millions of requests
3. **Reliability** - ACID transactions
4. **Querying** - Advanced analytics capabilities
5. **Concurrent Access** - Multiple instances can share data
6. **Backup/Restore** - Standard database backup tools

## Production Considerations

1. **Connection Pooling** - Already configured (pool_size=10, max_overflow=20)
2. **Indexes** - Added on frequently queried columns
3. **Soft Deletes** - API keys marked inactive instead of deleted
4. **Timestamps** - All records timestamped for auditing
5. **JSON Storage** - Request metadata stored for analytics

## Troubleshooting

### Connection Refused

```bash
# Check if PostgreSQL is running
docker-compose ps
# or
brew services list

# Check DATABASE_URL in .env file
```

### Migration Errors

```bash
# Drop and recreate database
docker-compose down -v
docker-compose up -d postgres

# Restart the app (tables will be recreated)
docker-compose up -d ollama-api
```

### Performance Issues

```sql
-- Check slow queries
SELECT * FROM pg_stat_statements ORDER BY mean_exec_time DESC;

-- Analyze table statistics
ANALYZE api_keys;
ANALYZE usage_logs;
```

## Next Steps

- Set up automated backups
- Configure replication for high availability
- Implement data archiving for old usage logs
- Add database monitoring and alerts
