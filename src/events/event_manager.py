from src.database.db import Session
from sqlalchemy import select, delete
from time import time

from src.database.models import Event
from src.exception_handlers.event_exception import EventNotFoundException


class EventManager:
    def __init__(self, session: Session):
        self.session = session
        self.active_events: dict[tuple[int, int, str], float] = {}

    def add_event(
        self,
        person_id: int,
        bag_id: int,
        reason: str,
        image_path: str,
        event_type: str
    ) -> Event | None:

        key = (person_id, bag_id, reason)
        now = time()

        cooldown = 10

        if key in self.active_events:
            if now - self.active_events[key] < cooldown:
                return None

        self.active_events[key] = now

        expired = []

        for key, ts in self.active_events.items():
            if now - ts > cooldown:
                expired.append(key)

        for key in expired:
            del self.active_events[key]

        new_event = Event(
            person_id=person_id,
            bag_id=bag_id,
            reason=reason,
            image_path=image_path,
            event_type=event_type
        )

        try:
            self.session.add(new_event)
            self.session.commit()
            self.session.refresh(new_event)

            return new_event

        except Exception as e:
            self.session.rollback()
            print(f"Event error: {e}")
            return None
        
    def get_events(self):
        result = self.session.execute(select(Event))

        return result.scalars().all()
    
    def get_event_by_id(self, event_id: int):
        result = self.session.execute(
            select(Event)
            .where(Event.id == event_id)
        )
        
        if not result:
            raise EventNotFoundException("Event not found")
        
        return result.scalar_one_or_none()
    
    def get_all_reasons(self):
        result = self.session.execute(
            select(Event.reason)
            .where(Event.reason.is_not(None))
            .distinct()
            .order_by(Event.reason)
        )

        return result.scalars().all()
    
    def get_all_event_type(self):
        result = self.session.execute(
            select(Event.event_type)
            .where(Event.event_type.is_not(None))
            .distinct()
            .order_by(Event.event_type)
        )

        return result.scalars().all()

    def get_events_by_reason(self, reason: str):
        result = self.session.execute(
            select(Event)
            .where(Event.reason == reason)
        )

        return result.scalars().all()
    
    def get_events_by_event_type(self, event_type: str):
        result = self.session.execute(
            select(Event)
            .where(Event.event_type == event_type)
        )

        return result.scalars().all()
        
    def delete_all_events(self):
        self.session.execute(delete(Event))
        self.session.commit()

        return {"detail": "All Events deleted"}