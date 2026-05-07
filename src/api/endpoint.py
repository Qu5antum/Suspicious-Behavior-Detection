from fastapi import APIRouter

from src.events.event_manager import EventManager
from src.database.db import session

event_router = APIRouter(
    prefix="/event",
    tags=["events"]
)


@event_router.get("/get_events", status_code=200)
async def get_events():
    event = EventManager(session=session)
    return event.get_events()