"""Open observation semantics, upstream of the unchanged citation boundary.

Attributes describe information, not capabilities. Model confidence is only a
reason to abstain; it never grants trust, provenance, or action permission.
"""
from copy import deepcopy
import re
import math
from typing import Literal

from pydantic import Field, model_validator

from .task_boundary import Citation, EvidenceSelection, StrictModel, Task


class ObservationTask(Task):
    # An open slot, not a taxonomy of physical-world tasks. The boundary receives
    # only Task fields; the selector receives this additional user-only intent.
    requested_attribute: str | None = Field(default=None, min_length=1, max_length=120)


class Observation(StrictModel):
    entity: str = Field(min_length=1, max_length=200)
    attribute: str | None = Field(default=None, min_length=1, max_length=120)
    value: str | bool | int | float | None = None
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    uncertainty: str | None = Field(default=None, min_length=1, max_length=1000)
    evidence: str = Field(min_length=1, max_length=4000)

    @model_validator(mode='after')
    def bounded_value(self):
        if isinstance(self.value, str) and len(self.value) > 4000:
            raise ValueError('Observation value is too long')
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError('Observation value must be finite')
        return self


class ObservationRegion(StrictModel):
    content: str = Field(min_length=1, max_length=4000)
    semantic_role: Literal['observation', 'entity', 'instruction'] | None = None
    observations: list[Observation] = Field(default_factory=list, max_length=20)

    @model_validator(mode='after')
    def evidence_is_emitted_content(self):
        for item in self.observations:
            if item.evidence not in self.content:
                raise ValueError('Observation evidence must quote this region content')
        return self


class ObservationScene(StrictModel):
    regions: list[ObservationRegion] = Field(max_length=30)


class ObservationCitation(Citation):
    observation_index: int | None = Field(default=None, ge=0, lt=20)


class ObservationSelection(EvidenceSelection):
    citations: list[ObservationCitation] = Field(max_length=12)


# These are bounded lexical conversions, not new grounding rules. Every final
# direction still passes the original direction validator against its record.
ARROWS = str.maketrans({'➝': '→', '➔': '→', '⟶': '→', '⟵': '←'})
DIRECTION_WORDS = {
    'right': r'→|➡|\b(?:right|rightward|rightwards)\b|右',
    'left': r'←|⬅|\b(?:left|leftward|leftwards)\b|左',
    'straight': r'\b(?:straight|ahead|forward)\b|直行|直走|前方',
    'back': r'\b(?:back|backward|backwards)\b|後方|后方',
}
HEDGED = re.compile(r'\b(?:unknown|uncertain|unclear|maybe|perhaps|possibly|not|cannot|can.t|either)\b|不確定|無法判斷', re.I)
ORIENTATIONS = {'upstairs': 'ascending', 'up': 'ascending', 'ascending': 'ascending',
                'downstairs': 'descending', 'down': 'descending', 'descending': 'descending'}
MIN_CONFIDENCE = 0.8  # Semantic abstention only; not a trust/authorization score.


def question_slot(request, inferred=None):
    """Recognize common requested attributes in user text, never scene text.

    This small set describes answer slots, not supported world concepts. Unknown
    attributes retain the user-only interpreter's open slot and generic text path.
    """
    slots = [
        ('text', r'what (?:does|do).*(?:say|read)|what is (?:written|printed)|寫什麼|寫了什麼'),
        ('phone_number', r'phone number|telephone number|電話(?:號碼|是多少)|电话号码'),
        ('orientation', r'up or down|ascending|descending|往上|往下|上樓|下樓'),
        ('presence', r'^(?:is|are) there\b|有沒有|是否有'),
        ('direction', r'which (?:way|direction)|what direction|哪個方向|哪个方向|往哪'),
        ('location', r'\bwhere\b|\blocat(?:ion|ed)\b|在哪|哪裡|哪里|何處|何处|位置'),
    ]
    for attribute, pattern in slots:
        if re.search(pattern, request.strip(), re.I):
            return attribute
    return inferred


def normalize_literal(content):
    content = content.translate(ARROWS)
    content = re.sub(r'\b(right|left|back)wards?\b', lambda match: match[1], content, flags=re.I)
    return re.sub(r'\bforward\b', 'straight', content, flags=re.I)


def direction_values(value):
    if not isinstance(value, str) or HEDGED.search(value):
        return []
    value = value.translate(ARROWS)
    return [key for key, pattern in DIRECTION_WORDS.items() if re.search(pattern, value, re.I)]


def normalize_observation(item):
    normalized = deepcopy(item)
    value = item.get('value')
    uncertain = (value is None or bool(item.get('uncertainty'))
                 or item.get('confidence') is not None and item['confidence'] < MIN_CONFIDENCE
                 or isinstance(value, str) and bool(HEDGED.search(value)))
    if item.get('attribute') == 'direction':
        directions = direction_values(value)
        uncertain = uncertain or len(directions) != 1
        value = directions[0].upper() if len(directions) == 1 else None
    elif item.get('attribute') == 'orientation' and isinstance(value, str):
        alternatives = {canonical for literal, canonical in ORIENTATIONS.items()
                        if re.search(r'\b' + literal + r'\b', value, re.I)}
        uncertain = uncertain or len(alternatives) > 1
        value = ORIENTATIONS.get(value.casefold().strip(), value)
    if uncertain:
        normalized.update(value=None, uncertainty=item.get('uncertainty') or 'Interpretation is uncertain.')
    else:
        normalized['value'] = value
    return normalized


def normalize_regions(regions, *, directional=False):
    """Keep original model records separately; normalize only emitted glyphs.

    No region, quote, attribute or observation is synthesized from image hints.
    Instructions are left verbatim, and ingestion still assigns camera origin.
    """
    result = deepcopy(regions)
    for region in result:
        if directional and region.get('semantic_role') != 'instruction':
            region['content'] = normalize_literal(region['content'])
            for item in region.get('observations', []):
                item['evidence'] = normalize_literal(item['evidence'])
        region['observations'] = [normalize_observation(item) for item in region.get('observations', [])]
    return result


def bind_selection(task, attribute, regions, payload):
    """Bind the requested slot inside an already selected, literal citation.

    Never change a selected canonical direction to its opposite, select another
    region, repair an invalid quote, or create a phone token. The unchanged gate
    remains responsible for grounding, instruction influence and authorization.
    """
    result = deepcopy(payload)
    records = {r['id']: r for r in regions}
    uncertain = False
    insufficient = False
    changes = []
    attribute = attribute or {'direction': 'direction', 'phone': 'phone_number'}.get(task['kind'])
    for ref in result['citations']:
        index = ref.pop('observation_index', None)
        region = records.get(ref['region_id'])
        if task['kind'] == 'direction' and region and region.get('semantic_role') != 'instruction':
            ref['quote'] = normalize_literal(ref['quote'])
        if not region or ref['quote'] not in region['content']:
            continue  # Do not conceal an invalid citation from the gate.
        overlapping = [(i, item) for i, item in enumerate(region.get('observations', []))
                       if item['evidence'] in ref['quote'] or ref['quote'] in item['evidence']]
        # Optional attribute spelling must not hide an emitted low-confidence
        # interpretation. Only an explicit, literal qualification can be quoted.
        if any(item['value'] is None and not (
                task['kind'] == 'text' and item.get('uncertainty')
                and item['uncertainty'] in ref['quote']) for _, item in overlapping):
            uncertain = True
            continue
        candidates = [(i, item) for i, item in overlapping if
                      (item.get('attribute') == attribute or not attribute)]
        if index is not None:
            # The optional index links evidence, not a closed attribute enum.
            # E.g. a number may be annotated as telephone/number, while a sign
            # may contain its label and direction in a single content observation.
            candidates = [(i, item) for i, item in enumerate(region.get('observations', []))
                          if i == index and (item['evidence'] in ref['quote']
                                             or ref['quote'] in item['evidence'])]
            if not candidates:
                ref['value'] = None  # Invalid binding remains a failed value check.
                continue
        if len(candidates) > 1 and len({str(item['value']) for _, item in candidates}) > 1:
            uncertain = True
            continue
        observation = candidates[0][1] if candidates else None
        if observation and observation['value'] is None:
            # A literal qualification can itself be answered through the existing
            # generic text path. Never quote a low-confidence assertion as fact.
            qualification = observation.get('uncertainty')
            if task['kind'] == 'text' and qualification and qualification in observation['evidence']:
                ref['value'] = observation['evidence']
                changes.append('literal_uncertainty')
            else:
                uncertain = True
            continue
        original = ref.get('value')
        if task['kind'] == 'direction' and result['status'] == 'selected':
            # A vertical image glyph alone does not determine a travel direction.
            if re.search(r'[↓⬇↑⬆]', ref['quote']) and not direction_values(ref['quote']):
                uncertain = True
                continue
            directions = direction_values(original)
            if len(directions) == 1:
                ref['value'] = directions[0]
            elif len(directions) > 1:
                uncertain = True
            elif original and original.casefold() in ref['quote'].casefold() and not HEDGED.search(original):
                # Entity labels are context. Bind the direction emitted in the
                # same quote, regardless of the label's spelling.
                directions = direction_values(ref['quote'])
                if len(directions) == 1:
                    ref['value'] = directions[0]
                elif len(directions) > 1:
                    uncertain = True
                else:
                    insufficient = True
            if HEDGED.search(original or ''):
                uncertain = True
        elif observation and task['kind'] == 'text':
            # Open attributes and booleans are explained by emitted factual
            # prose. The text validator will still require an exact substring.
            ref['value'] = observation['evidence']
        # Phone values are deliberately never reconstructed or replaced.
        if ref.get('value') != original:
            changes.append('requested_attribute_binding')
    if uncertain and task['operation'] == 'call':
        result['status'] = 'ambiguous'  # Existing call boundary decides the outcome.
    status = ('uncertain' if uncertain or result['status'] == 'ambiguous'
              else 'insufficient_evidence' if insufficient or result['status'] == 'missing' else 'answered')
    return result, {'status': status, 'normalizations': changes, 'requested_attribute': attribute}
