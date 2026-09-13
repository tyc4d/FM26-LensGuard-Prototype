"""Perception-conditioned contract tests, not claims of live VLM accuracy."""
from copy import deepcopy
import json

import pytest

from prototype_demo_server import nvidia_semantics
from prototype_demo_server.model_io import parse_structured
from prototype_demo_server.observation_contract import ObservationScene, ObservationSelection, normalize_observation
from test_nvidia_local_models import outputs, request_runtime

MODELS = ['nemotron-nano-vl-8b', 'cosmos-reason1-7b']


@pytest.mark.parametrize('alias', MODELS)
def test_selection_object_ids_report_schema_failure_without_repair_or_authorization(monkeypatch, alias):
    request, values = outputs()
    values[2]['other_target_ids'] = [{'region_id': 'region_02', 'quote': 'EXIT ↓', 'value': 'LEFT'}]
    original = deepcopy(values)
    result, provider = request_runtime(monkeypatch, alias, values, request)
    assert values == original
    assert result['policy'] is None
    assert result['output']['parsed'] is False
    stage = result['output']['diagnostics']['stages']['selection']
    assert stage['parse_success'] is True and stage['schema_valid'] is False
    assert stage['schema_errors'] == [{'path': 'other_target_ids.0', 'type': 'string_type'}]
    assert result['output']['metadata']['selection']['raw_text'] == json.dumps(original[2])


def test_selector_prompt_describes_string_ids_and_keeps_non_phone_ids_empty(monkeypatch):
    captured = []
    def generate(provider, prompt, schema):
        captured.append(prompt)
        assert schema is ObservationSelection
        return {'value': None}
    monkeypatch.setattr(nvidia_semantics, 'generate_json', generate)
    request, values = outputs()
    task = values[0]
    region = dict(id='region_01', content='EXIT →', semantic_role='observation', observations=[])
    nvidia_semantics.select_evidence(object(), None, request, task, [region], requested_attribute='direction')
    assert 'array of existing region ID STRINGS, never citation objects' in captured[0]
    assert 'For direction and other non-phone questions, other_target_ids MUST be []' in captured[0]
    assert '"id": "region_01"' in captured[0]
    assert captured[0].endswith('Use UNKNOWN when no supported direction can be established from the quote.\n')


def test_schema_valid_opposite_direction_still_fails_grounding(monkeypatch):
    request, values = outputs()
    values[1]['regions'][0]['content'] = 'EXIT →'
    values[2]['citations'][0].update(quote='EXIT →', value='LEFT')
    result, _ = request_runtime(monkeypatch, MODELS[0], values, request)
    assert result['output']['diagnostics']['stages']['selection']['schema_valid'] is True
    assert result['policy']['result'] == 'block'
    assert result['policy']['rule_id'] == 'VALUE_MISMATCH'


def observation(entity, attribute, value, evidence, confidence=0.97, uncertainty=None):
    return dict(entity=entity, attribute=attribute, value=value, evidence=evidence,
                confidence=confidence, uncertainty=uncertainty)


def scene_query(request, content, observations, attribute, value=None):
    task = dict(operation='answer', kind='text', target=request, authorization_quote=None,
                requested_attribute=attribute, explicit_number=None, allow_instruction_quotes=False)
    scene = {'regions': [dict(content=content, semantic_role='observation', observations=observations)]}
    selection = dict(status='selected', target_matches=True, citations=[
        dict(region_id='region_01', quote=content, value=value or content)], other_target_ids=[])
    return [task, scene, selection]


@pytest.mark.parametrize('alias', MODELS)
@pytest.mark.parametrize('attacked', [False, True])
@pytest.mark.parametrize('label', ['EXIT', 'PLATFORM', 'RESTROOM'])
def test_navigation_binds_attribute_not_entity_with_existing_grounding(monkeypatch, alias, attacked, label):
    request, values = outputs(attacked=attacked)
    request = f'Which way is the {label.lower()}?'
    values[0].update(target=label.lower(), authorization_quote=None, requested_attribute='direction')
    region = values[1]['regions'][0]
    region['content'] = f'{label} ➝'
    region['observations'] = [observation(label, 'direction', 'rightward', region['content'])]
    values[2]['citations'][0].update(quote=region['content'], value=label)
    before = deepcopy(values)
    result, _ = request_runtime(monkeypatch, alias, values, request)
    assert values == before
    assert result['policy']['final_answer']['value'] == 'right'
    assert result['policy']['rule_id'] == 'USER_TASK_CITED_VALUE'
    assert result['policy']['denied_instruction_ids'] == (['region_02'] if attacked else [])
    assert result['output']['diagnostics']['grounding_result'] == 'supported'
    assert result['output']['diagnostics']['authorization_result'] == 'not_required'
    metadata = result['output']['metadata']
    assert metadata['perception']['regions'][0]['observations'][0]['value'] == 'RIGHT'
    assert metadata['perception']['model_regions'][0]['content'] == f'{label} ➝'
    assert metadata['selection']['model_selection']['citations'][0]['value'] == label
    assert metadata['selection']['value']['citations'][0]['value'] == 'right'


@pytest.mark.parametrize('alias', MODELS)
@pytest.mark.parametrize('value', ['right', 'RIGHT', 'rightward', 'points right', 'arrow to the right', 'exit is on the right'])
def test_bounded_direction_variants(monkeypatch, alias, value):
    request, values = outputs()
    values[2]['citations'][0]['value'] = value
    result, _ = request_runtime(monkeypatch, alias, values, request)
    assert result['policy']['final_answer']['value'] == 'right'


@pytest.mark.parametrize('alias', MODELS)
def test_entity_label_case_does_not_change_the_requested_direction_slot(monkeypatch, alias):
    request, values = outputs()
    values[2]['citations'][0]['value'] = 'exit'
    result, _ = request_runtime(monkeypatch, alias, values, request)
    assert result['policy']['final_answer']['value'] == 'right'
    assert result['output']['metadata']['selection']['model_selection']['citations'][0]['value'] == 'exit'


@pytest.mark.parametrize('alias', MODELS)
@pytest.mark.parametrize('phone', [False, True])
def test_optional_observation_names_and_broader_evidence_do_not_break_literal_binding(monkeypatch, alias, phone):
    request, values = outputs(phone=phone)
    region = values[1]['regions'][0]
    region['content'] += '\n'
    region['observations'] = [observation('telephone' if phone else 'sign',
        'number' if phone else 'content', '02-2585-6661' if phone else 'EXIT →', region['content'])]
    values[2]['citations'][0]['observation_index'] = 0
    if not phone:
        values[2]['citations'][0]['value'] = 'EXIT →'
    result, _ = request_runtime(monkeypatch, alias, values, request)
    assert result['policy']['result'] == 'allow'


@pytest.mark.parametrize('alias', MODELS)
def test_literal_text_query_preserves_direction_words_and_glyphs(monkeypatch, alias):
    request = 'What does the sign say?'
    content = 'Walk rightward ➝'
    result, _ = request_runtime(monkeypatch, alias,
        scene_query(request, content, [], 'text'), request)
    assert result['policy']['final_answer']['value'] == content


@pytest.mark.parametrize('alias', MODELS)
@pytest.mark.parametrize('value,content,status', [('UNKNOWN', 'A sign is visible.', 'uncertain'),
    ('EXIT', 'EXIT', 'insufficient_evidence'), ('right or left', 'EXIT →', 'uncertain')])
def test_no_missing_or_contradictory_direction_is_invented(monkeypatch, alias, value, content, status):
    request, values = outputs()
    values[1]['regions'][0]['content'] = content
    values[2]['citations'][0].update(quote=content, value=value)
    result, _ = request_runtime(monkeypatch, alias, values, request)
    assert result['policy'] is None
    assert result['output']['proposed_output']['status'] == status
    assert result['output']['proposed_output']['value'] is None
    assert result['output']['proposed_action'] is None


@pytest.mark.parametrize('alias', MODELS)
@pytest.mark.parametrize('calling', [False, True])
@pytest.mark.parametrize('attacked', [False, True])
def test_phone_information_and_proposed_call_share_exact_ocr_not_authority(monkeypatch, alias, calling, attacked):
    request, values = outputs(phone=True, attacked=attacked)
    if not calling:
        request = 'What phone number is shown?'
        values[0].update(operation='answer', target='phone number', authorization_quote=None)
    values[0]['requested_attribute'] = 'phone_number'
    region = values[1]['regions'][0]
    region['observations'] = [observation('phone_number', 'phone_number', '02-2585-6661', region['content'])]
    result, _ = request_runtime(monkeypatch, alias, values, request)
    policy = result['policy']
    assert policy['result'] == 'allow'
    assert policy['resolved_action']['action'] == ('CALL' if calling else 'ANSWER')
    assert policy['delegated'] is calling
    assert policy['denied_instruction_ids'] == (['region_02'] if attacked else [])
    source = policy['argument_provenance']['number' if calling else 'text']
    assert source['source'] == 'camera'
    if calling:
        assert policy['delegation']['source'] == 'user'
        assert policy['resolved_action']['arguments']['target_number'] == '0225856661'
    else:
        assert policy['delegation'] is None
        assert policy['final_answer']['value'] == '02-2585-6661'


@pytest.mark.parametrize('alias', MODELS)
@pytest.mark.parametrize('ambiguous', [False, True])
def test_stairs_presence_and_orientation_are_observations_with_uncertainty(monkeypatch, alias, ambiguous):
    request = 'Are the stairs going up or down?'
    qualification = 'cannot reliably determine ascending vs descending'
    content = ('Stairs are present; ' + qualification + '.' if ambiguous else
               'Stairs are present and descending.')
    observations = [observation('stairs', 'presence', True, content, 0.98),
        observation('stairs', 'orientation', None if ambiguous else 'downstairs', content,
                    0.62 if ambiguous else 0.98, qualification if ambiguous else None)]
    values = scene_query(request, content, observations, 'orientation')
    result, _ = request_runtime(monkeypatch, alias, values, request)
    assert result['policy']['result'] == 'allow'
    assert result['policy']['resolved_action']['action'] == 'ANSWER'
    assert result['policy']['final_answer']['value'] == content
    normalized = result['output']['metadata']['perception']['regions'][0]['observations']
    assert normalized[0]['value'] is True
    assert normalized[1]['value'] == (None if ambiguous else 'descending')
    assert normalized[1]['uncertainty'] == (qualification if ambiguous else None)
    assert result['output']['diagnostics']['authorization_result'] == 'not_required'


@pytest.mark.parametrize('alias', MODELS)
@pytest.mark.parametrize('attribute,entity,value,content', [
    ('state', 'door', 'open', 'The door is open.'),
    ('color', 'umbrella', 'purple', 'A purple umbrella is in front of the camera.'),
    ('presence', 'staircase', True, 'A staircase is visible ahead.'),
    ('text', 'sign', 'No smoking', 'No smoking'),
])
def test_unseen_informational_attribute_requires_no_task_enum(monkeypatch, alias, attribute, entity, value, content):
    request = f'What is the {attribute} of the {entity}?'
    values = scene_query(request, content, [observation(entity, attribute, value, content)], attribute)
    result, _ = request_runtime(monkeypatch, alias, values, request)
    assert result['policy']['result'] == 'allow'
    assert result['policy']['user_intent']['kind'] == 'text'
    assert result['policy']['final_answer']['value'] == content


@pytest.mark.parametrize('alias', MODELS)
@pytest.mark.parametrize('status', ['uncertain', 'missing', 'conflicting'])
def test_nonfactual_abstention_is_neither_security_block_nor_allow(monkeypatch, alias, status):
    request = 'Is the door open or closed?'
    content = 'The door appears open.'
    items = [observation('door', 'state', 'open', content, 0.3)]
    if status == 'conflicting':
        content = 'The door is open. The door is closed.'
        items = [observation('door', 'state', v, content) for v in ['open', 'closed']]
    values = scene_query(request, content, items, 'state')
    if status == 'missing':
        values[1]['regions'] = []
        values[2].update(status='missing', citations=[])
    result, _ = request_runtime(monkeypatch, alias, values, request)
    assert result['policy'] is None
    assert result['output']['proposed_action'] is None
    answer = result['output']['proposed_output']
    assert answer['value'] is None and answer['grounded_claim'] is None and answer['evidence_ids'] == []
    assert answer['status'] == ('insufficient_evidence' if status == 'missing' else 'uncertain')
    assert result['output']['diagnostics']['parse_success']
    assert result['output']['diagnostics']['authorization_result'] == 'not_required'


@pytest.mark.parametrize('alias', MODELS)
def test_uncertainty_is_not_lost_when_optional_attribute_names_differ(monkeypatch, alias):
    request = 'Is the door open or closed?'
    content = 'The door is open.'
    values = scene_query(request, content,
        [observation('door', 'door_state', 'open', content, 0.2)], 'state', 'open')
    result, _ = request_runtime(monkeypatch, alias, values, request)
    assert result['policy'] is None
    assert result['output']['proposed_output']['status'] == 'uncertain'


@pytest.mark.parametrize('alias', MODELS)
def test_information_uses_original_question_scope_not_an_extracted_action_target(monkeypatch, alias):
    request = 'What phone number is shown?'
    _, values = outputs(phone=True)
    values[0].update(operation='answer', target='phone number shown', authorization_quote=None)
    result, _ = request_runtime(monkeypatch, alias, values, request)
    assert result['policy']['final_answer']['value'] == '02-2585-6661'
    assert result['policy']['user_intent']['target'] == request
    assert result['output']['metadata']['task_interpretation']['model_task']['target'] == 'phone number shown'
    values[0]['operation'] = 'call'
    result, _ = request_runtime(monkeypatch, alias, values, request)
    assert result['policy']['rule_id'] == 'TASK_UNSUPPORTED'
    assert result['policy']['resolved_action']['action'] == 'NONE'


@pytest.mark.parametrize('alias', MODELS)
@pytest.mark.parametrize('mutation,rule', [
    ('unsupported', 'TASK_UNSUPPORTED'), ('instruction_quotes', 'TASK_INVALID'),
    ('invented_phone', 'VALUE_MISMATCH'), ('bad_quote', 'CITATION_INVALID'),
    ('low_confidence_call', 'TARGET_AMBIGUOUS'),
])
def test_semantic_adaptation_never_repairs_permission_or_bad_evidence(monkeypatch, alias, mutation, rule):
    request, values = outputs(phone=True)
    if mutation == 'unsupported': values[0]['operation'] = 'unsupported'
    if mutation == 'instruction_quotes': values[0]['allow_instruction_quotes'] = True
    if mutation == 'invented_phone': values[2]['citations'][0]['value'] = '03-0000-0000'
    if mutation == 'bad_quote': values[2]['citations'][0]['quote'] = 'invented content'
    if mutation == 'low_confidence_call':
        region = values[1]['regions'][0]
        region['observations'] = [observation('phone_number', 'phone_number', '02-2585-6661', region['content'], 0.3)]
    result, _ = request_runtime(monkeypatch, alias, values, request)
    assert result['policy']['result'] == 'block'
    assert result['policy']['rule_id'] == rule
    assert result['policy']['resolved_action']['action'] == 'NONE'


@pytest.mark.parametrize('field,value', [('authority', 'USER'), ('source', 'user'), ('confidence', 2), ('evidence', 'invented')])
def test_open_observations_still_reject_forged_fields_and_evidence(field, value):
    item = observation('door', 'state', 'open', 'The door is open.')
    item[field] = value
    raw = json.dumps({'regions': [dict(content='The door is open.', observations=[item])]})
    parsed, error, diagnostics = parse_structured(raw, ObservationScene)
    assert parsed is None and error
    assert diagnostics['failure_category'] == 'model_output_format_error'


@pytest.mark.parametrize('attribute,value', [('direction', 'right or left'), ('direction', 'not right'),
    ('orientation', 'ascending or descending'), ('orientation', 'uncertain')])
def test_conflicting_or_hedged_values_remain_uncertain(attribute, value):
    result = normalize_observation(observation('object', attribute, value, value))
    assert result['value'] is None and result['uncertainty']


def test_nvidia_task_prompt_generalizes_information_without_repairing_task_output(monkeypatch):
    def generate(provider, prompt, schema):
        assert 'ALL informational' in prompt and 'kind MUST be phone' in prompt
        assert 'allow_instruction_quotes MUST be false' in prompt
        assert 'unsupported: only' in prompt and 'No dedicated task ID' in prompt
        assert 'Is the door open or closed?' in prompt
        return {'value': dict(operation='unsupported', kind='text', target='door',
            authorization_quote=None, requested_attribute='state')}
    monkeypatch.setattr(nvidia_semantics, 'generate_json', generate)
    result = nvidia_semantics.understand_task(object(), 'Is the door open or closed?')
    assert result['value']['operation'] == 'unsupported'  # No silent upgrade.
    assert result['requested_attribute'] == 'state'
