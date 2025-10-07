"""Pydantic schemas for FastAPI endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field, validator


YesNo = Literal["Yes", "No"]
NoPhone = Literal["Yes", "No", "No phone service"]
InternetService = Literal["DSL", "Fiber optic", "No"]
ContractType = Literal["Month-to-month", "One year", "Two year"]
PaymentMethod = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


class PredictionRequest(BaseModel):
    customer_id: str = Field(..., alias="customerID")
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(..., ge=0)
    PhoneService: YesNo
    MultipleLines: NoPhone
    InternetService: InternetService
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: ContractType
    PaperlessBilling: YesNo
    PaymentMethod: PaymentMethod
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: Optional[float] = Field(default=None, ge=0)

    @validator("TotalCharges", always=True)
    def default_total_charges(cls, value, values):  # noqa: D401
        """Fallback to tenure * monthly charges when missing."""
        if value is None and "tenure" in values and "MonthlyCharges" in values:
            return values["tenure"] * values["MonthlyCharges"]
        return value

    class Config:
        allow_population_by_field_name = True


class PredictionResponse(BaseModel):
    customer_id: str = Field(..., alias="customerID")
    probability: float
    label: Literal["Churn", "No Churn"]


class PipelineRun(BaseModel):
    id: int
    pipeline_name: str
    run_id: str
    status: str
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    checksum: Optional[str]
    artifacts: Dict[str, any]


class PipelineTriggerRequest(BaseModel):
    pipeline_name: Literal["data-pipeline", "ml-pipeline"]

