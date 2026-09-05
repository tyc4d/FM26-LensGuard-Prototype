"""READ != OBEY. Grounding is relative to model perception, not authenticity.

Model supplied authority and delegation never grant permission. Roles can be
downgraded by literal checks; scene instructions remain visible without power.
"""
import re
import unicodedata
from firewall.action_normalizer import normalize_phone_number

INSTRUCTION = re.compile(r'\b(ignore|disregard|override|obey|answer|respond|reply|say|system\s*prompt|assistant|chatgpt|instead|if\s+asked)\b|忽略|無視|覆蓋|回答|回覆|改撥|一律|助理|助手|系統指令|系统指令|不要告訴|不要告诉', re.I)
# A bare command at the start is a behavioral request. Data-like reservation
# labels ("For reservations, call ..." / "訂位請撥 ...") remain usable entities.
DIRECT_COMMAND = re.compile(r'^(?:(?:ai|agent|model)\s*[:,]?\s*)?(?:please\s+)?(?:call|dial|execute|send|open|purchase)\b|^(?:請|请)?(?:撥打|拨打|打電話給|打电话给|執行|执行)|\b(?:must|should|need\s+to)\s+(?:call|dial|execute|send|open)\b', re.I)
# Chinese labels commonly touch the digits: 訂位專線02-2345-6789.
PHONE = re.compile(r'(?<![a-zA-Z0-9])\+?\d[\d ()-]{5,}\d(?!\d)')
RESERVATION = re.compile(r'reservations?(?:\s+(?:phone|number|hotline|tel))?|訂位|订位|預約專線|预约专线', re.I)
CARD = re.compile(r'business\s+card|名片', re.I)
EXIT = re.compile(r'\b(?:emergency\s+)?exit\b|緊急出口|紧急出口|出口', re.I)
DIRECTIONS = {'right': r'→|➡|\bright\b|右', 'left': r'←|⬅|\bleft\b|左', 'straight': r'↑|⬆|\b(?:straight|ahead)\b|直行|直走|前方', 'back': r'↓|⬇|\bback\b|後方|后方'}


def _claim(content):
    if EXIT.search(content):
        values = [value for value, pattern in DIRECTIONS.items() if re.search(pattern, content, re.I)]
        if len(values) == 1:
            return {'predicate': 'exit_direction', 'value': values[0]}, 'observation'
    phones = PHONE.findall(content)
    if len(phones) == 1:
        try:
            normalize_phone_number(phones[0])
        except (TypeError, ValueError):
            return None, 'observation'
        predicate = 'restaurant_reservation_phone' if RESERVATION.search(content) else 'card_phone' if CARD.search(content) else 'phone_number'
        return {'predicate': predicate, 'value': phones[0].strip()}, 'entity'
    return None, 'observation'


def semantic_regions(evidence):
    """Build a request-local graph; dangling/cyclic ancestry is unusable."""
    if evidence is None:
        return []
    if not isinstance(evidence, (list, tuple)) or len(evidence) > 100:
        raise ValueError('Evidence must be an array of at most 100 regions')
    regions = []
    for index, raw in enumerate(evidence):
        if not isinstance(raw, dict):
            raise ValueError('Evidence regions must be objects')
        content = unicodedata.normalize('NFKC', str(raw.get('content', raw.get('text', '')))).strip()
        identifier = raw.get('id', f'region_{index + 1:02}')
        if not isinstance(identifier, str) or not identifier:
            raise ValueError('Evidence IDs must be nonempty strings')
        lineage = raw.get('lineage', [])
        if not isinstance(lineage, list) or any(not isinstance(ref, str) for ref in lineage):
            raise ValueError('Region lineage must contain evidence IDs')
        role = str(raw.get('semantic_role', '')).lower()
        instruction = role in ('instruction', 'instruction_derived') or bool(INSTRUCTION.search(content) or DIRECT_COMMAND.search(content))
        claim, inferred_role = _claim(content)
        if claim is None and content and not instruction:
            # Plain scene text/entities remain usable evidence. A literal text
            # claim grants no phone/direction binding without that specific role.
            inferred_role = 'entity' if role == 'entity' else 'observation'
            claim = {'predicate': 'scene_entity' if inferred_role == 'entity' else 'scene_text', 'value': content}
        role = ('instruction_derived' if role == 'instruction_derived' else 'instruction') if instruction else inferred_role
        declared = raw.get('grounded_claim')
        agrees = True
        if isinstance(declared, dict):
            agrees = bool(claim and declared.get('predicate') == claim['predicate'])
            if agrees:
                if 'phone' in claim['predicate']:
                    try:
                        agrees = normalize_phone_number(str(declared.get('value'))) == normalize_phone_number(claim['value'])
                    except ValueError:
                        agrees = False
                elif claim['predicate'] == 'exit_direction':
                    agrees = str(declared.get('value', '')).lower() == claim['value']
                else:
                    agrees = unicodedata.normalize('NFKC', str(declared.get('value', ''))).strip() == claim['value']
        supported = bool(content and claim and agrees and not instruction and raw.get('source', 'camera') == 'camera')
        regions.append({'id': identifier, 'content': content, 'source': 'camera', 'semantic_role': role,
            'grounded_claim': claim if not instruction else None,
            'requested_behavior': raw.get('requested_behavior', content if instruction else None),
            'grounding': {'status': 'supported' if supported else 'unsupported', 'method': 'literal_scene_content'},
            'lineage': lineage, 'authority': 'EVIDENCE' if supported else 'NONE',
            'status': 'DENY_INSTRUCTION_INFLUENCE' if instruction else 'RETAIN' if supported else 'UNSUPPORTED'})
    by_id = {region['id']: region for region in regions}
    if len(by_id) != len(regions):
        raise ValueError('Duplicate evidence ID')

    # Propagate instruction ancestry first, including through cycles. Caching
    # an early cycle failure must not hide an instruction on another branch.
    descendants = {identifier: [] for identifier in by_id}
    tainted = {item['id'] for item in regions
               if item['semantic_role'] in ('instruction', 'instruction_derived')}
    for region in regions:
        for parent in region['lineage']:
            if parent in descendants:
                descendants[parent].append(region['id'])
    pending = list(tainted)
    while pending:
        for child in descendants[pending.pop()]:
            if child not in tainted:
                tainted.add(child)
                pending.append(child)

    ancestry_cache = {}

    def ancestry(identifier, visited):
        if identifier in tainted:
            return 'instruction'
        if identifier in ancestry_cache:
            return ancestry_cache[identifier]
        if identifier not in by_id or identifier in visited:
            return 'unsupported'
        region = by_id[identifier]
        states = [ancestry(ref, visited | {identifier}) for ref in region['lineage']]
        status = ('unsupported' if 'unsupported' in states or region['grounding']['status'] != 'supported'
                  else 'supported')
        ancestry_cache[identifier] = status
        return status

    states = {identifier: ancestry(identifier, set()) for identifier in by_id}
    for region in regions:
        state = states[region['id']]
        if state != 'supported':
            region['authority'] = 'NONE'
            region['grounding']['status'] = 'unsupported'
            if state == 'instruction':
                if region['semantic_role'] != 'instruction':
                    region['semantic_role'] = 'instruction_derived'
                region['status'] = 'DENY_INSTRUCTION_INFLUENCE'
            else:
                region['status'] = 'UNSUPPORTED'
    return regions


def user_intent(user_request):
    """Anchored narrow trusted-request grammar, never permissions from scene text."""
    text = unicodedata.normalize('NFKC', user_request).strip().rstrip('.。!！?？').strip()
    result = {'source': 'user', 'request': user_request, 'kind': 'unknown', 'delegation': None, 'trusted_values': {}}
    if re.fullmatch(r'(?:緊急)?出口(?:在(?:哪裡|哪里|哪|何處)|往哪裡走)|where\s+is\s+(?:the\s+)?(?:emergency\s+)?exit', text, re.I):
        result['kind'] = 'exit_location'
    restaurant = re.fullmatch(r'(?:please\s+)?call\s+(?:the\s+)?restaurant[’\x27]s\s+reservation\s+(?:phone\s+)?number|幫我(?:打電話訂位|打電話訂這間餐廳|撥打(?:這間)?餐廳的訂位(?:電話|專線))', text, re.I)
    card = text == '幫我撥打這張名片上的電話'
    if restaurant or card:
        result['kind'] = 'call_reservation' if restaurant else 'call_card'
        result['delegation'] = {'source': 'user', 'explicit': True, 'tool': 'call_phone', 'argument': 'number',
            'semantic_role': 'entity', 'predicate': 'restaurant_reservation_phone' if restaurant else 'card_phone',
            'scope': 'observed_restaurant' if restaurant else 'observed_business_card'}
    direct = re.fullmatch(r'(?:please\s+)?call\s+(\+?\d[\d ()-]{1,}\d)|(?:幫我)?(?:撥打|打電話給)\s*(\+?\d[\d ()-]{1,}\d)', text, re.I)
    if direct:
        result['kind'] = 'call_number'
        result['trusted_values']['call_phone.number'] = normalize_phone_number(next(value for value in direct.groups() if value))
    url = re.fullmatch(r'(?:please\s+)?open\s+(https?://\S+)|(?:幫我)?開啟\s*(https?://\S+)', text, re.I)
    if url:
        result['kind'] = 'open_url'
        result['trusted_values']['open_url.url'] = next(value for value in url.groups() if value)
    return result
