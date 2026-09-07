"""Model-agnostic demo transport/format diagnostics, with no policy decisions."""
import json
import re
from time import perf_counter

from providers.local.base_local_vlm import (
    PreparedLocalInput, extract_single_json_object, input_token_count, move_inputs_to_device,
)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f'Nonstandard JSON constant: {value}')


def parse_structured(raw, schema, *, collection_field=None):
    """Accept one final object, optionally normalizing its collection envelope.

    Never repair values, references or authority.

    A complete leading reasoning block is debug-only; it cannot supply JSON to
    the schema. Truncated/repeated wrappers and duplicate keys are format errors.
    Existing balanced-object/fence extraction remains the inner parser.
    """
    diagnostics = {'parse_success': False, 'schema_valid': False,
                   'failure_category': None, 'normalization_method': None}
    try:
        candidate = raw.strip()
        if candidate.startswith('<think>'):
            match = re.fullmatch(r'<think>(.*?)</think>\s*(.+)', candidate, re.S)
            if not match or '<think>' in match[1] or '</think>' in match[2]:
                raise ValueError('Incomplete or repeated reasoning wrapper; no final result')
            candidate = match[2].strip()
            diagnostics['normalization_method'] = 'complete_reasoning_wrapper_removed'
        if candidate.startswith('<answer>'):
            match = re.fullmatch(r'<answer>\s*(.*?)\s*</answer>', candidate, re.S)
            if not match or '<answer>' in match[1] or '</answer>' in match[1]:
                raise ValueError('Incomplete or repeated final-answer wrapper')
            candidate = match[1]
            diagnostics['normalization_method'] = 'complete_final_answer_wrapper'
        if candidate.startswith(('<think>', '</think>', '</answer>')):
            raise ValueError('Unexpected output wrapper')
        if collection_field is not None:
            # Only the declared collection envelope may be normalized. Literal
            # entries are unchanged and the full strict schema still validates.
            unfenced = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', candidate, re.S | re.I)
            envelope = unfenced[1].strip() if unfenced else candidate
            if envelope.startswith('['):
                items = json.loads(envelope, object_pairs_hook=_unique_object,
                                   parse_constant=_reject_constant)
                if not isinstance(items, list):
                    raise ValueError('Collection output must be an array')
                if len(items) == 1 and isinstance(items[0], dict) and set(items[0]) == {collection_field}:
                    payload = items[0]
                    method = 'single_collection_envelope_unwrapped'
                else:
                    payload = {collection_field: items}
                    method = 'declared_collection_envelope_added'
                candidate = json.dumps(payload, ensure_ascii=False)
                diagnostics['normalization_method'] = method
        extracted, _ = extract_single_json_object(candidate)
        payload = json.loads(extracted, object_pairs_hook=_unique_object)
        diagnostics['parse_success'] = True
        value = schema.model_validate(payload).model_dump()
        diagnostics['schema_valid'] = True
        return value, None, diagnostics
    except (ValueError, TypeError, AttributeError) as exc:
        diagnostics['failure_category'] = 'model_output_format_error'
        diagnostics['error_message'] = str(exc)
        return None, str(exc), diagnostics


def generate(provider, prompt, image=None, *, move_inputs=move_inputs_to_device):
    started = perf_counter()
    if image is not None:
        prepared = provider._prepare_input(prompt, image)
    elif callable(getattr(provider, '_prepare_text_input', None)):
        prepared = provider._prepare_text_input(prompt)
    else:
        # Preserve the existing Gemma/Qwen text transport exactly.
        inputs = provider.processor.apply_chat_template(
            [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt')
        inputs = move_inputs(inputs, provider.device, dtype=provider._torch_module().bfloat16)
        prepared = PreparedLocalInput(payload=inputs, input_token_count=input_token_count(inputs),
                                      metadata={'chat_template_adapter': 'demo-text-only-v1'})
    provider._synchronize()
    preprocessing_ms = (perf_counter() - started) * 1000
    started = perf_counter()
    with provider._torch_module().inference_mode():
        result = provider._generate(prepared)
    provider._synchronize()
    return result.raw_text, {
        'preprocessing_ms': preprocessing_ms, 'generation_ms': (perf_counter() - started) * 1000,
        'input_tokens': getattr(prepared, 'input_token_count', None),
        'output_tokens': getattr(result, 'output_token_count', None),
        'adapter': getattr(prepared, 'metadata', {}),
        'generation': getattr(result, 'metadata', {}),
    }


def outcome_diagnostics(diagnostics, policy):
    """Label existing outcomes without altering a gate decision or claiming OCR truth."""
    result = {**diagnostics, 'perception_correctness': 'not_independently_verified',
              'grounding_result': 'not_evaluated', 'authorization_result': 'not_evaluated'}
    if policy is None:
        if (not result.get('failure_category') and result.get('parse_success')
                and result.get('informational_status') in {'uncertain', 'insufficient_evidence'}):
            result.update(failure_category=('model_uncertainty' if result['informational_status'] == 'uncertain'
                                           else 'evidence_unavailable'),
                          authorization_result='not_required')
        return result
    rule = policy['rule_id']
    if policy['result'] == 'allow':
        result['grounding_result'] = 'supported'
        result['authorization_result'] = (
            'not_required' if policy['use'] == 'INFORMATIONAL_OUTPUT' else 'authorized')
    elif rule in {'CITATION_INVALID', 'VALUE_MISMATCH', 'TARGET_UNRESOLVED', 'TARGET_AMBIGUOUS'}:
        result.update(grounding_result='failed', failure_category='grounding_failure')
    elif rule == 'EVIDENCE_MISSING':
        result.update(grounding_result='missing', failure_category='evidence_unavailable')
    elif rule in {'TASK_UNAVAILABLE', 'TASK_UNSUPPORTED', 'TASK_INVALID', 'SELECTION_INVALID'}:
        result['failure_category'] = 'task_or_selection_unavailable'
    elif rule in {'INSTRUCTION_SELECTED', 'TASK_ACTION_MISMATCH', 'USER_VALUE_INVALID'}:
        result.update(authorization_result='denied', failure_category='authorization_denial')
    else:
        result['failure_category'] = 'unclassified_policy_result'
    return result
