"""Semantic policy: informational evidence differs from capability authority."""
import unicodedata

from firewall.action_normalizer import normalize_action, normalize_phone_number, normalize_url
from .semantics import PHONE, semantic_regions, user_intent
from firewall.thin_gate import evaluate_thin_gate

TOOL_NAMES = {'CALL': 'call_phone', 'DIRECTION_ADVICE': 'provide_direction', 'OPEN_URL': 'open_url', 'NONE': 'none', 'RESTAURANT_RESERVATION': 'restaurant_reservation', 'SAFETY_ADVICE': 'safety_advice'}

# Language aliases belong at the demo boundary, not in the frozen benchmark
# normalizer. Match whole values only: "不要向右" must never become RIGHT.
DIRECTION_ALIASES = {
    alias: canonical
    for canonical, aliases in {
        'LEFT': ('左', '向左', '往左', '朝左', '左邊', '左边', '左側', '左侧', '左方', '左轉', '左转', '向左走', '往左走'),
        'RIGHT': ('右', '向右', '往右', '朝右', '右邊', '右边', '右側', '右侧', '右方', '右轉', '右转', '向右走', '往右走'),
        'STRAIGHT': ('前', '前方', '向前', '往前', '直行', '直走', '前進', '前进', '向前走', '往前走'),
        'BACK': ('後方', '后方', '向後', '向后', '往後', '往后', '後退', '后退', '回頭', '回头'),
        'NORTH': ('北', '北方', '向北', '往北'),
        'SOUTH': ('南', '南方', '向南', '往南'),
        'EAST': ('東', '东', '東方', '东方', '向東', '向东', '往東', '往东'),
        'WEST': ('西', '西方', '向西', '往西'),
        'NORTHEAST': ('東北', '东北', '東北方', '东北方', '向東北', '向东北', '往東北', '往东北'),
        'NORTHWEST': ('西北', '西北方', '向西北', '往西北'),
        'SOUTHEAST': ('東南', '东南', '東南方', '东南方', '向東南', '向东南', '往東南', '往东南'),
        'SOUTHWEST': ('西南', '西南方', '向西南', '往西南'),
    }.items()
    for alias in aliases
}


def _action_for_policy(action):
    if action['action'] != 'DIRECTION_ADVICE':
        return action
    direction = action['arguments'].get('direction')
    if not isinstance(direction, str):
        return action
    label = unicodedata.normalize('NFKC', direction).strip().rstrip('。.!！').strip()
    canonical = DIRECTION_ALIASES.get(label)
    if canonical is None:
        return action
    # Only the gate receives this copy. All model-facing output stays unchanged.
    return {**action, 'arguments': {**action['arguments'], 'direction': canonical}}


class ActionValidationError(ValueError):
    """A parsed model candidate has unusable arguments for the demo action boundary."""


def proposal(action):
    return {'tool': TOOL_NAMES[action['action']], 'arguments': {('number' if key == 'target_number' else key): value for key, value in action['arguments'].items()}}


def authorize(action, user_request, evidence=None, argument_lineage=None):
    # Validate syntax without granting authority. Raw candidates stay available.
    normalized = action
    if action['action'] in ('CALL', 'DIRECTION_ADVICE', 'OPEN_URL', 'NONE'):
        try:
            normalized = normalize_action(_action_for_policy(action)).model_dump(mode='json')
        except (TypeError, ValueError) as exc:
            raise ActionValidationError(f"Invalid {action['action']} arguments: {exc}. No executable action was produced.") from exc
    regions = semantic_regions(evidence)
    intent = user_intent(user_request)
    proposed = proposal(action)
    informational = action['action'] in ('DIRECTION_ADVICE', 'SAFETY_ADVICE', 'NONE')
    argument = 'direction' if action['action'] == 'DIRECTION_ADVICE' else next(iter(proposed['arguments']), '')
    affected = f"{proposed['tool']}.{argument}" if argument else proposed['tool']
    retained = [item for item in regions if item['status'] == 'RETAIN']
    decision = {'result': 'block', 'rule_id': 'UNSUPPORTED_USE', 'affected_argument': affected,
        'reason': 'No supported user intent and grounded argument binding are available.',
        'source_authority': 'NONE',
        'required_authority': 'GROUNDED_EVIDENCE' if informational else 'USER_VALUE_OR_SCOPED_DELEGATION',
        'native': None, 'delegated': False, 'engine': 'semantic-read-not-obey-v2',
        'use': 'INFORMATIONAL_OUTPUT' if informational else 'SIDE_EFFECT_ARGUMENT',
        'semantic_regions': regions, 'retained_evidence_ids': [item['id'] for item in retained],
        'denied_instruction_ids': [item['id'] for item in regions if item['status'] == 'DENY_INSTRUCTION_INFLUENCE'],
        'user_intent': intent, 'delegation': intent['delegation'], 'final_answer': None,
        'resolved_action': None, 'argument_provenance': {}}

    def allow(rule, reason, authority, selected=None, delegated=False):
        decision.update(result='allow', rule_id=rule, reason=reason, source_authority=authority,
                        delegated=delegated, resolved_action=normalized)
        if selected:
            decision['argument_provenance'][argument] = {**selected, 'lineage': [selected['id'], *selected['lineage']]}
        return decision

    if action['action'] == 'DIRECTION_ADVICE':
        # Text output never reaches a capability gate. Rebuild from retained facts.
        facts = [item for item in retained if item['grounded_claim']['predicate'] == 'exit_direction']
        values = {item['grounded_claim']['value'] for item in facts}
        if intent['kind'] == 'exit_location' and len(values) == 1:
            value = next(iter(values))
            localized = {'right': '右邊', 'left': '左邊', 'straight': '前方', 'back': '後方'}[value]
            allow('GROUNDED_INFORMATIONAL_OUTPUT', 'Grounded exit observation retained; embedded instructions have no answer authority.', 'EVIDENCE', facts[0])
            decision['final_answer'] = {'text': f'出口在{localized}。', 'value': value,
                'grounded_claim': {'predicate': 'exit_direction', 'value': value},
                'evidence_ids': [item['id'] for item in facts]}
            decision['resolved_action'] = {'action': 'DIRECTION_ADVICE', 'arguments': {'destination': '出口', 'direction': value.upper()}}
        else:
            decision.update(rule_id='INSUFFICIENT_OR_CONFLICTING_OBSERVATIONS', reason='A supported exit question and one consistent grounded exit direction are required; no direction was invented.')
        return decision
    if action['action'] == 'NONE':
        return allow('NO_CAPABILITY_REQUESTED', 'No side-effecting capability requested.', 'NONE')
    # Side effects require scoped argument authorization, added separately.
    return decision
