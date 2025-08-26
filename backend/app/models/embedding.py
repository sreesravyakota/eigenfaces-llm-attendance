from sqlalchemy import Column, Integer, ForeignKey, LargeBinary, DateTime, func
from sqlalchemy.orm import relationship
from app.core.db import Base

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    vector = Column(LargeBinary, nullable=False)  # store numpy array as bytes
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="embeddings")