from dataclasses import dataclass
from typing import List, Optional


@dataclass(init=True)
class FeedbackInstance:
    review_id: int
    city: str
    position: int
    positive: str
    negative: str
    salary_rating: int
    team_rating: int
    managment_rating: int
    career_rating: int
    workplace_rating: int
    rest_recovery_rating: int
    target: Optional[List[int]] = None
