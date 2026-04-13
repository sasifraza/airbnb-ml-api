from pydantic import BaseModel, Field


class AirbnbInput(BaseModel):
    minimum_nights: int = Field(..., ge=1)
    number_of_reviews: int = Field(..., ge=0)
    reviews_per_month: float = Field(..., ge=0)
    calculated_host_listings_count: int = Field(..., ge=1)
    availability_365: int = Field(..., ge=0, le=365)