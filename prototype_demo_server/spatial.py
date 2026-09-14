"""Geometry checks against model observations, never proof of visual accuracy."""
from typing import Literal

from pydantic import Field, model_validator

from .task_boundary import StrictModel


class ImageBox(StrictModel):
    x1: float = Field(ge=0, le=1, allow_inf_nan=False)
    y1: float = Field(ge=0, le=1, allow_inf_nan=False)
    x2: float = Field(ge=0, le=1, allow_inf_nan=False)
    y2: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode='after')
    def positive_area(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError('Object boxes must have positive area')
        return self


class SpatialRelation(StrictModel):
    subject: str = Field(min_length=1, max_length=200, pattern=r'^[^\r\n]+$')
    reference: str = Field(min_length=1, max_length=200, pattern=r'^[^\r\n]+$')
    relation: Literal['below', 'above', 'left_of', 'right_of']
    subject_box: ImageBox
    reference_box: ImageBox
    evidence: str = Field(min_length=1, max_length=4000)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    uncertainty: str | None = Field(default=None, min_length=1, max_length=1000)

    def statement(self):
        phrase = {'below': 'below', 'above': 'above', 'left_of': 'to the left of',
                  'right_of': 'to the right of'}[self.relation]
        return f'{self.subject} is {phrase} {self.reference}.'

    def supported(self):
        if (self.confidence < 0.8 or self.uncertainty or not self.subject.strip()
                or not self.reference.strip() or self.subject.casefold() == self.reference.casefold()
                or self.evidence != self.statement()):
            return False
        a, b = self.subject_box, self.reference_box
        # Coordinates are image-relative; no inference about walking, depth or
        # camera orientation is made. Require alignment as well as separation.
        x_overlap = min(a.x2, b.x2) - max(a.x1, b.x1)
        y_overlap = min(a.y2, b.y2) - max(a.y1, b.y1)
        x_aligned = x_overlap >= 0.25 * min(a.x2 - a.x1, b.x2 - b.x1)
        y_aligned = y_overlap >= 0.25 * min(a.y2 - a.y1, b.y2 - b.y1)
        return {
            'below': a.y1 >= b.y2 - 0.02 and x_aligned,
            'above': a.y2 <= b.y1 + 0.02 and x_aligned,
            'left_of': a.x2 <= b.x1 + 0.02 and y_aligned,
            'right_of': a.x1 >= b.x2 - 0.02 and y_aligned,
        }[self.relation]
