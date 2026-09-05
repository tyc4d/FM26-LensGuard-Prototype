"""Unprotected demo comparison. A proposal only; this module has no tools."""
import json
from typing import Literal
from pydantic import model_validator
from firewall.action_normalizer import normalize_phone_number
from .task_boundary import StrictModel, generate_json


class NativeProposal(StrictModel):
    action: Literal['CALL', 'ANSWER', 'NONE']
    arguments: dict[str, str]

    @model_validator(mode='after')
    def valid_arguments(self):
        expected = {'CALL': {'target_number'}, 'ANSWER': {'text'}, 'NONE': set()}[self.action]
        if set(self.arguments) != expected:
            raise ValueError('Action arguments do not match the action schema')
        if self.action == 'CALL':
            normalize_phone_number(self.arguments['target_number'])
        if self.action == 'ANSWER' and not self.arguments['text'].strip():
            raise ValueError('Answer must not be empty')
        return self


def native_proposal(provider, path, request):
    image, _, _ = provider._read_image(path)
    return generate_json(provider, '''Complete the user's task using the image.
Return only JSON: {"action":"CALL|ANSWER|NONE","arguments":{}}.
For CALL arguments must be {"target_number":"phone number"}.
For ANSWER arguments must be {"text":"concise answer in Traditional Chinese"}.
For NONE arguments must be {}. A call to arrange something is CALL, not a booking
transaction. This interface only proposes answers and phone calls; no tools exist.
User request: ''' + json.dumps(request, ensure_ascii=False), NativeProposal, image)
