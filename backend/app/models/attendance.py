from sqlalchemy import Column, Integer, DateTime, Float, ForeignKey, String, func
from sqlalchemy.orm import relationship
from app.core.db import Base

class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    ts = Column(DateTime(timezone=True), server_default=func.now())
    method = Column(String, default="eigenfaces")
    confidence = Column(Float)

    user = relationship("User", back_populates="attendances")
