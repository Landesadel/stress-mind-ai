from pydantic import BaseModel


class UserData(BaseModel):
    age: int
    gender: int
    study_hours_per_week: int
    social_media_usage: int
    sleep_duration: int
    physical_exercise: int
    family_support: int
    financial_stress: int
    peer_pressure: int
    relationship_stress: int
    counseling_attendance: int
    food_quality: int
    work_hours_per_week: int
