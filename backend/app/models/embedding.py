from sqlalchemy import (
    Column, Integer, ForeignKey, LargeBinary, DateTime, func, String,
    UniqueConstraint, Index
)
from sqlalchemy.orm import relationship
from app.core.db import Base

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    __table_args__ = (
        # one embedding per (user, method): e.g., (1,'eigenfaces') and (1,'nn')
        UniqueConstraint("user_id", "method", name="uq_user_method"),
        Index("ix_face_embeddings_method", "method"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # which pipeline produced this vector
    method = Column(String, nullable=False, default="eigenfaces")

    # dimensionality of the vector 
    dim = Column(Integer, nullable=False, default=0)

    vector = Column(LargeBinary, nullable=False)  # numpy bytes
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="embeddings")
