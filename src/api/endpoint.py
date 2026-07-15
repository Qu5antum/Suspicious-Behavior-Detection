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


@event_router.get("/get_reasons", status_code=200)
async def get_reasons():
    return event.get_all_reasons()


@event_router.get("/get_event_types", status_code=200)
async def get_event_types():
    return event.get_all_event_type()


@event_router.get("/get_events_by_type", status_code=200)
async def get_event_by_type(
    event_type: str
):
    return event.get_events_by_event_type(event_type=event_type)


@event_router.delete("/delete_events", status_code=200)
async def delete_events():
    return event.delete_all_events()