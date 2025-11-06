from pydantic import BaseModel, Field
from typing import Optional
import yaml

class BlurSettings(BaseModel):
    min_degree: int = Field(30, ge=1)
    max_degree: int = Field(90)
    min_angle: float = 0
    max_angle: float = 180
    intensity_factor: float = 1.5
    color_shift: bool = True
    blend_original: float = Field(0.6, ge=0.0, le=1.0)
    curved_motion: bool = True
    noise_stddev: float = 4.0

class Config(BaseModel):
    input_dir: str
    output_dir: str
    blur: BlurSettings

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)
