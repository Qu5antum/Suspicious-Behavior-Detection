from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy import create_engine

engine = create_engine(
    f"sqlite:///database.db",
    connect_args={"check_same_thread": False}
)

Base = declarative_base()

session = sessionmaker(
    engine, class_=Session, expire_on_commit=False
)

def get_session() -> Session:
    with session as session:
        yield session