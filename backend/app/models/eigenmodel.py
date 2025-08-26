from sqlalchemy import Column, Integer, LargeBinary, DateTime, func
from app.core.db import Base

class EigenModel(Base):
    __tablename__ = "eigen_model"

    id = Column(Integer, primary_key=True, default=1) 
    mean = Column(LargeBinary, nullable=False)        
    components = Column(LargeBinary, nullable=False) 
    n_components = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
