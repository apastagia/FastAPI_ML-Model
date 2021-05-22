from pydantic import BaseModel

#class which describe bank notes measurements
class BankNote(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float