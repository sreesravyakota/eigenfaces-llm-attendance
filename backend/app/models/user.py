from sqlalchemy import Column, Integer, String, DateTime, func
from app.core.db import Base
from sqlalchemy.orm import relationship


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    attendances = relationship("Attendance", back_populates="user")
    embeddings = relationship("FaceEmbedding", back_populates="user")