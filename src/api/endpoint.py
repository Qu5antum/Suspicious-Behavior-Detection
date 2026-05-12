from fastapi import APIRouter

from src.events.event_manager import EventManager
from src.database.db import session

event_router = APIRouter(
    prefix="/event",
    tags=["events"]
)

event = EventManager(session=session)


@event_router.get("/get_events", status_code=200)
async def get_events():
    return event.get_events()


@event_router.delete("/delete_events", status_code=200)
async def delete_events():
    return event.delete_all_events()