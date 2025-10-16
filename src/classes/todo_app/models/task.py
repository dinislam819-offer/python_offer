from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

class Task(BaseModel):
    model_config = ConfigDict()
    
    id: int
    title: str = Field(..., min_length=3, max_length=100)
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    category_id: int
    status_id: int
    
    # Optional fields for different task behaviors
    is_done: Optional[bool] = None  # For simple tasks
    deadline: Optional[datetime] = None  # For urgent tasks
    repeat_every: Optional[str] = None  # For recurring tasks ("day", "week", "month")

    def model_post_init(self, __context) -> None:
        """Set default values for optional fields"""
        if self.is_done is None:
            self.is_done = False