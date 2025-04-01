from pydantic import BaseModel


class AdviceResponse(BaseModel):
    advice: str
