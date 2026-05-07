from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy import create_engine
from contextlib import contextmanager


engine = create_engine(
    f"sqlite:///database.db",
    connect_args={"check_same_thread": False}
)

Base = declarative_base()

SessionLocal = sessionmaker(
    engine, class_=Session, expire_on_commit=False
)

session = SessionLocal()


@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()