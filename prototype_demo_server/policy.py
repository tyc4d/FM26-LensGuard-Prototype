"""Deterministic demo boundary. No fabricated semantic/region provenance."""
import unicodedata

from firewall.action_normalizer import normalize_action
from firewall.thin_gate import evaluate_thin_gate

TOOL_NAMES = {'CALL': 'call_phone', 'DIRECTION_ADVICE': 'navigate', 'OPEN_URL': 'open_url', 'NONE': 'none', 'RESTAURANT_RESERVATION': 'restaurant_reservation', 'SAFETY_ADVICE': 'safety_advice'}

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


def authorize(action, user_request):
    # The frozen parser validates string types, not whether a direction/target
    # can be normalized. Check that boundary before either policy or delegation.
    # Keep the normalized copy for the gate; retain the original for display.
    gate_action = action
    if action['action'] in ('CALL', 'DIRECTION_ADVICE', 'OPEN_URL', 'NONE'):
        try:
            gate_action = normalize_action(_action_for_policy(action))
        except (TypeError, ValueError) as exc:
            raise ActionValidationError(
                f"Invalid {action['action']} arguments: {exc}. No executable action was produced."
            ) from exc
    proposed = proposal(action)
    argument = 'direction' if action['action'] == 'DIRECTION_ADVICE' else next(iter(proposed['arguments']), '')
    affected = f"{proposed['tool']}.{argument}" if argument else proposed['tool']
    # Exact, trusted task scope only; scenario IDs and VLM text never grant authority.
    # This allows the *proposed* card number for one simulated CALL, not proof of
    # its visual grounding/authenticity. No general natural-language delegation NLP.
    delegated = user_request.strip() == '幫我撥打這張名片上的電話' and action['action'] == 'CALL'
    native = None
    if delegated:
        result, rule = 'allow', 'DEMO_SCOPED_CARD_CALL_DELEGATION_V1'
        reason = 'Explicit user delegation permits the proposed card number for this simulated call; visual grounding remains unverified.'
    elif action['action'] in ('CALL', 'DIRECTION_ADVICE', 'OPEN_URL', 'NONE'):
        native = evaluate_thin_gate(gate_action, {'action': action['action'], 'arguments': {}, 'evidence_complete': False}).model_dump(mode='json')
        result = 'allow' if native['decision'] == 'ALLOW' else 'block'
        rule = native['policy_rules_triggered'][0]
        reason = native['user_message']
        if native['decision'] != 'ALLOW':
            reason += ' Automatic execution is blocked pending confirmation; no semantic evidence registry is available.'
    else:
        result, rule, reason = 'block', 'DEMO_UNSUPPORTED_POLICY_V1', 'No live authorization policy is available for this action. Execution is blocked.'
    return {'result': result, 'rule_id': rule, 'affected_argument': affected, 'reason': reason,
            'source_authority': 'DELEGATED' if delegated else 'OBSERVATION_ONLY',
            'required_authority': 'EXTERNAL_ACTION_TARGET', 'native': native,
            'delegated': delegated, 'engine': 'prototype-thin-gate-with-scoped-demo-delegation'}
