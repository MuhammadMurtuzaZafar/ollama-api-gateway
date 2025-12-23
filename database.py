"""
Database configuration and models for PostgreSQL
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, DateTime, Text, JSON
from datetime import datetime
import os
from dotenv import load_dotenv
import logging

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Database URL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/ollama_api"
)

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=True,  # Set to False in production
    pool_size=10,
    max_overflow=20
)

# Create async session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base class for models
class Base(DeclarativeBase):
    pass

# Database Models
class APIKey(Base):
    __tablename__ = "api_keys"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)  # Unique name constraint
    role: Mapped[str] = mapped_column(String(50), nullable=False, default="user")
    rate_limit: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)

class UsageLog(Base):
    __tablename__ = "usage_logs"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    api_key: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    endpoint: Mapped[str] = mapped_column(String(255), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    response_time: Mapped[float] = mapped_column(nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    request_data: Mapped[dict] = mapped_column(JSON, nullable=True)

# Dependency to get database session
async def get_db():
    """Dependency to get database session"""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()

# Database initialization
async def init_db():
    """Initialize database tables"""
    try:
        logger.info(f"Connecting to database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'local'}")
        async with engine.begin() as conn:
            logger.info("Creating database tables...")
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
        raise

async def close_db():
    """Close database connections"""
    logger.info("Closing database connections...")
    await engine.dispose()
    logger.info("Database connections closed")
