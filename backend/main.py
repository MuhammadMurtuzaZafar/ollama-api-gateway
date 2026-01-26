"""
EdgeLLM API Service with Authentication, Authorization, and Monitoring
High-performance LLM inference with deterministic latency for edge devices.
"""
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import jwt
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import logging
from contextlib import asynccontextmanager
from sqlalchemy import select, func, and_, delete
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db, init_db, close_db, APIKey, UsageLog, async_session_maker

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
EDGELLM_BASE_URL = os.getenv("EDGELLM_BASE_URL", "http://localhost:8080")

# Security
security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    logger.info("Starting EdgeLLM API Service...")
    try:
        await init_db()
        logger.info("Database initialized - tables created successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
        raise
    
    # Initialize demo API keys
    try:
        await initialize_demo_keys()
    except Exception as e:
        logger.error(f"Failed to initialize demo keys: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down EdgeLLM API Service...")
    await close_db()
    logger.info("Database connections closed")

# Initialize FastAPI app
app = FastAPI(
    title="EdgeLLM API Service",
    description="High-performance LLM inference with deterministic latency for edge devices. Features BitNet 1.58-bit quantization and 15.5x lower jitter than traditional inference.",
    version="1.0.0",
    dopenapi_version="3.0.3",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class User(BaseModel):
    username: str
    role: str = Field(default="user", description="User role: admin or user")

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class APIKeyCreate(BaseModel):
    name: str = Field(..., description="Name/description for this API key")
    role: str = Field(default="user", description="Role: admin or user")
    rate_limit: int = Field(default=100, description="Requests per hour")

class APIKeyResponse(BaseModel):
    api_key: str
    name: str
    role: str
    rate_limit: int
    created_at: str

class GenerateRequest(BaseModel):
    model: str = Field(..., description="Model name (e.g., llama2, mistral)")
    prompt: str = Field(..., description="The prompt to generate from")
    stream: bool = Field(default=False, description="Stream the response")
    temperature: Optional[float] = Field(default=0.7, description="Temperature for generation")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    model: str = Field(..., description="Model name")
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    stream: bool = Field(default=False, description="Stream the response")
    temperature: Optional[float] = Field(default=0.7, description="Temperature for generation")

class UsageStats(BaseModel):
    total_requests: int
    requests_by_model: Dict[str, int]
    requests_by_endpoint: Dict[str, int]
    average_response_time: float
    last_24h_requests: int

class DetailedStats(BaseModel):
    api_key_name: str
    api_key_role: str
    total_requests: int
    total_requests_24h: int
    total_requests_7d: int
    requests_by_model: Dict[str, int]
    requests_by_endpoint: Dict[str, int]
    average_response_time: float
    min_response_time: float
    max_response_time: float
    first_request: Optional[str]
    last_request: Optional[str]
    rate_limit: int
    rate_limit_usage_percent: float

# Helper Functions
async def initialize_demo_keys():
    """Initialize demo API keys from environment variables"""
    async with async_session_maker() as session:
        demo_keys = []
        
        # Load admin key from environment
        admin_key = os.getenv("DEMO_ADMIN_KEY")
        if admin_key:
            # Check if key already exists
            result = await session.execute(
                select(APIKey).where(APIKey.key == admin_key)
            )
            if not result.scalar_one_or_none():
                demo_keys.append(APIKey(
                    key=admin_key,
                    name="Demo Admin Key",
                    role="admin",
                    rate_limit=1000,
                    created_at=datetime.utcnow()
                ))
        
        # Load user key from environment
        user_key = os.getenv("DEMO_USER_KEY")
        if user_key:
            # Check if key already exists
            result = await session.execute(
                select(APIKey).where(APIKey.key == user_key)
            )
            if not result.scalar_one_or_none():
                demo_keys.append(APIKey(
                    key=user_key,
                    name="Demo User Key",
                    role="user",
                    rate_limit=100,
                    created_at=datetime.utcnow()
                ))
        
        if demo_keys:
            session.add_all(demo_keys)
            await session.commit()
            logger.info(f"Initialized {len(demo_keys)} demo API keys from environment")
        else:
            logger.warning("No new demo API keys to add. Set DEMO_ADMIN_KEY and DEMO_USER_KEY to enable demo keys.")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> APIKey:
    """Verify API key from Authorization header"""
    api_key = credentials.credentials
    
    # Query database for API key
    result = await db.execute(
        select(APIKey).where(
            and_(APIKey.key == api_key, APIKey.is_active == True)
        )
    )
    key_data = result.scalar_one_or_none()
    
    if not key_data:
        logger.warning(f"Invalid API key attempt: {api_key[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    # Check rate limiting
    if not await check_rate_limit(db, api_key, key_data.rate_limit):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    return key_data

async def check_rate_limit(db: AsyncSession, api_key: str, limit: int) -> bool:
    """Check if API key is within rate limit"""
    one_hour_ago = datetime.utcnow() - timedelta(hours=1)
    
    # Count requests from last hour
    result = await db.execute(
        select(func.count(UsageLog.id)).where(
            and_(
                UsageLog.api_key == api_key,
                UsageLog.timestamp > one_hour_ago
            )
        )
    )
    recent_count = result.scalar()
    
    return recent_count < limit

async def log_request(
    db: AsyncSession,
    api_key: str,
    endpoint: str,
    model: str,
    response_time: float,
    request_data: Optional[Dict[str, Any]] = None
):
    """Log API request for monitoring"""
    usage_log = UsageLog(
        api_key=api_key,
        endpoint=endpoint,
        model=model,
        response_time=response_time,
        timestamp=datetime.utcnow(),
        request_data=request_data
    )
    db.add(usage_log)
    await db.commit()

async def require_admin(key_data: APIKey = Depends(verify_api_key)) -> APIKey:
    """Require admin role"""
    if key_data.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return key_data

# API Endpoints

@app.get("/", tags=["General"])
async def root():
    """Root endpoint - API information"""
    return {
        "service": "EdgeLLM API Service",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs",
        "health": "/health",
        "features": ["BitNet 1.58-bit quantization", "15.5x lower jitter", "Edge deployment ready"]
    }

@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{EDGELLM_BASE_URL}/health", timeout=5.0)
            edgellm_status = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception as e:
        logger.error(f"EdgeLLM health check failed: {e}")
        edgellm_status = "unreachable"

    return {
        "status": "healthy",
        "edgellm_backend": edgellm_status,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/keys", response_model=APIKeyResponse, tags=["API Keys"])
async def create_api_key(
    key_request: APIKeyCreate,
    admin: APIKey = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Create a new API key (admin only)"""
    import secrets
    from sqlalchemy.exc import IntegrityError
    
    # Check if name already exists
    result = await db.execute(
        select(APIKey).where(APIKey.name == key_request.name)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"API key with name '{key_request.name}' already exists"
        )
    
    # Generate secure API key
    api_key = f"edgellm-{secrets.token_urlsafe(32)}"
    
    new_key = APIKey(
        key=api_key,
        name=key_request.name,
        role=key_request.role,
        rate_limit=key_request.rate_limit,
        created_at=datetime.utcnow()
    )
    
    try:
        db.add(new_key)
        await db.commit()
        await db.refresh(new_key)
        
        logger.info(f"Created new API key: {key_request.name} (role: {key_request.role})")
        
        return APIKeyResponse(
            api_key=api_key,
            name=new_key.name,
            role=new_key.role,
            rate_limit=new_key.rate_limit,
            created_at=new_key.created_at.isoformat()
        )
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"API key with name '{key_request.name}' already exists"
        )

@app.get("/api/keys", tags=["API Keys"])
async def list_api_keys(
    admin: APIKey = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """List all API keys (admin only)"""
    result = await db.execute(select(APIKey))
    api_keys = result.scalars().all()
    
    keys_list = []
    for key in api_keys:
        keys_list.append({
            "id": key.id,
            "key_preview": f"{key.key[:20]}...",
            "name": key.name,
            "role": key.role,
            "rate_limit": key.rate_limit,
            "created_at": key.created_at.isoformat(),
            "is_active": key.is_active
        })
    
    return {"api_keys": keys_list, "total": len(keys_list)}

@app.get("/api/keys/{key_id}/reveal", tags=["API Keys"])
async def reveal_api_key(
    key_id: int,
    admin: APIKey = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Reveal full API key (admin only) - Use with caution"""
    result = await db.execute(
        select(APIKey).where(APIKey.id == key_id)
    )
    key = result.scalar_one_or_none()
    
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")
    
    logger.warning(f"Admin revealed API key: {key.name} (ID: {key_id})")
    
    return {
        "id": key.id,
        "name": key.name,
        "key": key.key,
        "role": key.role,
        "rate_limit": key.rate_limit,
        "created_at": key.created_at.isoformat()
    }

@app.delete("/api/keys/{key_preview}", tags=["API Keys"])
async def revoke_api_key(
    key_preview: str,
    admin: APIKey = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Revoke an API key (admin only)"""
    # Find key that starts with preview
    result = await db.execute(select(APIKey))
    api_keys = result.scalars().all()
    
    for key in api_keys:
        if key.key.startswith(key_preview):
            # Soft delete - mark as inactive
            key.is_active = False
            await db.commit()
            logger.info(f"Revoked API key: {key_preview}")
            return {"message": "API key revoked successfully"}
    
    raise HTTPException(status_code=404, detail="API key not found")

@app.get("/api/models", tags=["EdgeLLM"])
async def list_models(
    key_data: APIKey = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db)
):
    """List available EdgeLLM models"""
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{EDGELLM_BASE_URL}/api/tags")
            response.raise_for_status()
            
            response_time = time.time() - start_time
            await log_request(db, key_data.key, "/api/models", "list", response_time)
            
            return response.json()
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

@app.post("/api/generate", tags=["EdgeLLM"])
async def generate(
    request: GenerateRequest,
    key_data: APIKey = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db)
):
    """Generate text using EdgeLLM with deterministic latency"""
    start_time = time.time()
    
    try:
        payload = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": request.stream,
        }
        
        if request.temperature is not None:
            payload["options"] = {"temperature": request.temperature}
        
        if request.max_tokens is not None:
            if "options" not in payload:
                payload["options"] = {}
            payload["options"]["num_predict"] = request.max_tokens
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{EDGELLM_BASE_URL}/api/generate",
                json=payload
            )
            response.raise_for_status()
            
            response_time = time.time() - start_time
            await log_request(
                db, key_data.key, "/api/generate", request.model, response_time,
                {"prompt_length": len(request.prompt)}
            )
            
            return response.json()
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")

@app.post("/api/chat", tags=["EdgeLLM"])
async def chat(
    request: ChatRequest,
    key_data: APIKey = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db)
):
    """Chat with EdgeLLM model"""
    start_time = time.time()
    
    try:
        messages = [msg.dict() for msg in request.messages]
        
        payload = {
            "model": request.model,
            "messages": messages,
            "stream": request.stream,
        }
        
        if request.temperature is not None:
            payload["options"] = {"temperature": request.temperature}
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{EDGELLM_BASE_URL}/api/chat",
                json=payload
            )
            response.raise_for_status()
            
            response_time = time.time() - start_time
            await log_request(
                db, key_data.key, "/api/chat", request.model, response_time,
                {"message_count": len(messages)}
            )
            
            return response.json()
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")

@app.get("/api/stats", response_model=UsageStats, tags=["Monitoring"])
async def get_usage_stats(
    key_data: APIKey = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db)
):
    """Get usage statistics for your API key"""
    # Get all logs for this API key
    result = await db.execute(
        select(UsageLog).where(UsageLog.api_key == key_data.key)
    )
    logs = result.scalars().all()
    
    if not logs:
        return UsageStats(
            total_requests=0,
            requests_by_model={},
            requests_by_endpoint={},
            average_response_time=0.0,
            last_24h_requests=0
        )
    
    # Calculate statistics
    total_requests = len(logs)
    
    requests_by_model = {}
    requests_by_endpoint = {}
    total_response_time = 0.0
    
    now = datetime.utcnow()
    last_24h = now - timedelta(hours=24)
    last_24h_count = 0
    
    for log in logs:
        # Count by model
        model = log.model or "unknown"
        requests_by_model[model] = requests_by_model.get(model, 0) + 1
        
        # Count by endpoint
        endpoint = log.endpoint or "unknown"
        requests_by_endpoint[endpoint] = requests_by_endpoint.get(endpoint, 0) + 1
        
        # Sum response times
        total_response_time += log.response_time
        
        # Count last 24h requests
        if log.timestamp > last_24h:
            last_24h_count += 1
    
    avg_response_time = total_response_time / total_requests if total_requests > 0 else 0.0
    
    return UsageStats(
        total_requests=total_requests,
        requests_by_model=requests_by_model,
        requests_by_endpoint=requests_by_endpoint,
        average_response_time=round(avg_response_time, 3),
        last_24h_requests=last_24h_count
    )

@app.get("/api/stats/detailed", response_model=DetailedStats, tags=["Monitoring"])
async def get_detailed_stats(
    key_data: APIKey = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed usage statistics for your API key with PostgreSQL aggregations"""
    # Use PostgreSQL aggregation functions for better performance
    from sqlalchemy import func as sql_func
    
    # Get aggregate stats
    stats_query = select(
        sql_func.count(UsageLog.id).label('total_requests'),
        sql_func.avg(UsageLog.response_time).label('avg_response_time'),
        sql_func.min(UsageLog.response_time).label('min_response_time'),
        sql_func.max(UsageLog.response_time).label('max_response_time'),
        sql_func.min(UsageLog.timestamp).label('first_request'),
        sql_func.max(UsageLog.timestamp).label('last_request')
    ).where(UsageLog.api_key == key_data.key)
    
    result = await db.execute(stats_query)
    stats = result.one()
    
    # Get 24h stats
    last_24h = datetime.utcnow() - timedelta(hours=24)
    count_24h = await db.execute(
        select(sql_func.count(UsageLog.id)).where(
            and_(
                UsageLog.api_key == key_data.key,
                UsageLog.timestamp > last_24h
            )
        )
    )
    requests_24h = count_24h.scalar() or 0
    
    # Get 7d stats
    last_7d = datetime.utcnow() - timedelta(days=7)
    count_7d = await db.execute(
        select(sql_func.count(UsageLog.id)).where(
            and_(
                UsageLog.api_key == key_data.key,
                UsageLog.timestamp > last_7d
            )
        )
    )
    requests_7d = count_7d.scalar() or 0
    
    # Get requests by model
    model_stats = await db.execute(
        select(
            UsageLog.model,
            sql_func.count(UsageLog.id).label('count')
        ).where(
            UsageLog.api_key == key_data.key
        ).group_by(UsageLog.model)
    )
    requests_by_model = {row.model or "unknown": row.count for row in model_stats}
    
    # Get requests by endpoint
    endpoint_stats = await db.execute(
        select(
            UsageLog.endpoint,
            sql_func.count(UsageLog.id).label('count')
        ).where(
            UsageLog.api_key == key_data.key
        ).group_by(UsageLog.endpoint)
    )
    requests_by_endpoint = {row.endpoint or "unknown": row.count for row in endpoint_stats}
    
    # Calculate rate limit usage
    rate_limit_usage = (requests_24h / key_data.rate_limit * 100) if key_data.rate_limit > 0 else 0
    
    return DetailedStats(
        api_key_name=key_data.name,
        api_key_role=key_data.role,
        total_requests=stats.total_requests or 0,
        total_requests_24h=requests_24h,
        total_requests_7d=requests_7d,
        requests_by_model=requests_by_model,
        requests_by_endpoint=requests_by_endpoint,
        average_response_time=round(stats.avg_response_time or 0.0, 3),
        min_response_time=round(stats.min_response_time or 0.0, 3),
        max_response_time=round(stats.max_response_time or 0.0, 3),
        first_request=stats.first_request.isoformat() if stats.first_request else None,
        last_request=stats.last_request.isoformat() if stats.last_request else None,
        rate_limit=key_data.rate_limit,
        rate_limit_usage_percent=round(rate_limit_usage, 2)
    )

@app.get("/api/admin/stats", tags=["Monitoring"])
async def get_all_stats(
    admin: APIKey = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get usage statistics for all API keys (admin only)"""
    all_stats = {}
    
    # Get all API keys
    result = await db.execute(select(APIKey))
    api_keys = result.scalars().all()
    
    for key in api_keys:
        # Get logs for this key
        logs_result = await db.execute(
            select(UsageLog).where(UsageLog.api_key == key.key)
        )
        logs = logs_result.scalars().all()
        
        all_stats[key.key[:20] + "..."] = {
            "key_name": key.name,
            "role": key.role,
            "total_requests": len(logs),
            "last_request": logs[-1].timestamp.isoformat() if logs else None,
            "is_active": key.is_active
        }
    
    return {"statistics": all_stats}


@app.get("/api/admin/stats/users", tags=["Monitoring"])
async def get_user_stats(
    admin: APIKey = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed user statistics for admin dashboard (admin only)"""
    from sqlalchemy import func as sql_func
    
    # Get all API keys
    result = await db.execute(select(APIKey).where(APIKey.is_active == True))
    api_keys = result.scalars().all()
    
    now = datetime.utcnow()
    last_24h = now - timedelta(hours=24)
    last_7d = now - timedelta(days=7)
    last_hour = now - timedelta(hours=1)
    
    user_stats = []
    
    for key in api_keys:
        # Get total requests
        total_result = await db.execute(
            select(sql_func.count(UsageLog.id)).where(UsageLog.api_key == key.key)
        )
        total_requests = total_result.scalar() or 0
        
        # Get 24h requests
        requests_24h_result = await db.execute(
            select(sql_func.count(UsageLog.id)).where(
                and_(
                    UsageLog.api_key == key.key,
                    UsageLog.timestamp > last_24h
                )
            )
        )
        requests_24h = requests_24h_result.scalar() or 0
        
        # Get 7d requests
        requests_7d_result = await db.execute(
            select(sql_func.count(UsageLog.id)).where(
                and_(
                    UsageLog.api_key == key.key,
                    UsageLog.timestamp > last_7d
                )
            )
        )
        requests_7d = requests_7d_result.scalar() or 0
        
        # Get hourly requests for rate limit
        requests_hour_result = await db.execute(
            select(sql_func.count(UsageLog.id)).where(
                and_(
                    UsageLog.api_key == key.key,
                    UsageLog.timestamp > last_hour
                )
            )
        )
        requests_hour = requests_hour_result.scalar() or 0
        
        # Get average response time
        avg_time_result = await db.execute(
            select(sql_func.avg(UsageLog.response_time)).where(UsageLog.api_key == key.key)
        )
        avg_response_time = avg_time_result.scalar() or 0
        
        # Get last request time
        last_request_result = await db.execute(
            select(sql_func.max(UsageLog.timestamp)).where(UsageLog.api_key == key.key)
        )
        last_request = last_request_result.scalar()
        
        # Get requests by model for this user
        model_stats = await db.execute(
            select(
                UsageLog.model,
                sql_func.count(UsageLog.id).label('count')
            ).where(
                UsageLog.api_key == key.key
            ).group_by(UsageLog.model)
        )
        requests_by_model = {row.model or "unknown": row.count for row in model_stats}
        
        # Get requests by endpoint for this user
        endpoint_stats = await db.execute(
            select(
                UsageLog.endpoint,
                sql_func.count(UsageLog.id).label('count')
            ).where(
                UsageLog.api_key == key.key
            ).group_by(UsageLog.endpoint)
        )
        requests_by_endpoint = {row.endpoint or "unknown": row.count for row in endpoint_stats}
        
        rate_limit_usage = (requests_hour / key.rate_limit * 100) if key.rate_limit > 0 else 0
        
        user_stats.append({
            "id": key.id,
            "name": key.name,
            "role": key.role,
            "key_preview": f"{key.key[:20]}...",
            "total_requests": total_requests,
            "requests_24h": requests_24h,
            "requests_7d": requests_7d,
            "requests_this_hour": requests_hour,
            "avg_response_time": round(avg_response_time, 3),
            "rate_limit": key.rate_limit,
            "rate_limit_usage": round(rate_limit_usage, 2),
            "last_request": last_request.isoformat() if last_request else None,
            "requests_by_model": requests_by_model,
            "requests_by_endpoint": requests_by_endpoint,
            "created_at": key.created_at.isoformat()
        })
    
    # Sort by total requests (highest first)
    user_stats.sort(key=lambda x: x['total_requests'], reverse=True)
    
    # Calculate global stats
    total_requests_all = sum(u['total_requests'] for u in user_stats)
    total_requests_24h_all = sum(u['requests_24h'] for u in user_stats)
    total_requests_7d_all = sum(u['requests_7d'] for u in user_stats)
    avg_response_time_all = sum(u['avg_response_time'] for u in user_stats) / len(user_stats) if user_stats else 0
    
    return {
        "users": user_stats,
        "summary": {
            "total_users": len(user_stats),
            "total_requests": total_requests_all,
            "total_requests_24h": total_requests_24h_all,
            "total_requests_7d": total_requests_7d_all,
            "avg_response_time": round(avg_response_time_all, 3)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
