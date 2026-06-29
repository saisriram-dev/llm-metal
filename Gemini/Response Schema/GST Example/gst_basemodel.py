from pydantic import BaseModel
from datetime import date
from typing import List


class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    amount: float


class Details(BaseModel):
    vendor: str
    date: date
    amount: float
    gst_number: str
    line_items: List[LineItem]
