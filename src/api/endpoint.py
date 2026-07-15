from fastapi import APIRouter

from src.events.event_manager import EventManager
from src.database.db import session

event_router = APIRouter(
    prefix="/api",
    tags=["events"]
)

event = EventManager(session=session)


@event_router.get("/event/events", status_code=200)
async def get_events():
    return event.get_events()

@event_router.get("/event/reasons", status_code=200)
async def get_reasons():
    return event.get_all_reasons()

@event_router.get("/event/types", status_code=200)
async def get_event_types():
    return event.get_all_event_type()

@event_router.get("/event/reason", status_code=200)
async def get_event_by_reason(
    reason: str
):
    return event.get_events_by_reason(reason=reason)

@event_router.get("/event/type", status_code=200)
async def get_event_by_type(
    event_type: str
):
    return event.get_events_by_event_type(event_type=event_type)

@event_router.get("/event/{event_id}", status_code=200)
async def get_event_by_id(
    event_id: int
):
    return event.get_event_by_id(event_id=event_id)

@event_router.delete("/event/delete_all", status_code=200)
async def delete_events():
    return event.delete_all_events()