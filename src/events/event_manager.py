from sqlalchemy.orm import Session

from src.database.models import Event

from time import time
from sqlalchemy.orm import Session

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


        

        