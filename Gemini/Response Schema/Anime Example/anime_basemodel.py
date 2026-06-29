from pydantic import BaseModel


class Anime(BaseModel):
    name: str
    status: str
    seasons: int
    episodes: int
