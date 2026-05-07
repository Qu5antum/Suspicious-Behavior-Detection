from src.database.db import engine
from src.database.models import Base

Base.metadata.create_all(bind=engine)