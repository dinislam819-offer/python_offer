from pydantic import BaseModel, Field

class Category(BaseModel):
    id: int
    name: str = Field(..., min_length=2, max_length=50)