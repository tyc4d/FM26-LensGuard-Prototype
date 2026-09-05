"""Deterministic demo boundary. No fabricated semantic/region provenance."""
from firewall.action_normalizer import normalize_action
from firewall.thin_gate import evaluate_thin_gate

TOOL_NAMES = {'CALL': 'call_phone', 'DIRECTION_ADVICE': 'navigate', 'OPEN_URL': 'open_url', 'NONE': 'none', 'RESTAURANT_RESERVATION': 'restaurant_reservation', 'SAFETY_ADVICE': 'safety_advice'}


class ActionValidationError(ValueError):
    """A parsed model candidate has unusable arguments for the demo action boundary."""


def proposal(action):
    return {'tool': TOOL_NAMES[action['action']], 'arguments': {('number' if key == 'target_number' else key): value for key, value in action['arguments'].items()}}


def authorize(action, user_request):
    # The frozen parser validates string types, not whether a direction/target
    # can be normalized. Check that boundary before either policy or delegation.
    # Discard the normalized copy: retain exactly the model's parsed arguments.
    if action['action'] in ('CALL', 'DIRECTION_ADVICE', 'OPEN_URL', 'NONE'):
        try:
            normalize_action(action)
        except (TypeError, ValueError) as exc:
            raise ActionValidationError(
                f"Invalid {action['action']} arguments: {exc}. No executable action was produced."
            ) from exc
    proposed = proposal(action)
    argument = next(iter(proposed['arguments']), '')
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
        native = evaluate_thin_gate(action, {'action': action['action'], 'arguments': {}, 'evidence_complete': False}).model_dump(mode='json')
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
