"""Spatial contracts are conditioned on perception, not proof of image accuracy."""
import pytest

from prototype_demo_server.observation_contract import question_slot, direction_values
from prototype_demo_server.semantics import DIRECTIONS
from test_nvidia_local_models import outputs, request_runtime


@pytest.mark.parametrize('query', ['Where is the emergency exit?', '出口在哪裡？', '出口的位置？'])
def test_location_question_is_not_forced_into_travel_direction(query):
    assert question_slot(query, 'direction') == 'location'


@pytest.mark.parametrize('query', ['Which way should I go?', 'Which direction is the exit?', '出口往哪走？'])
def test_travel_question_keeps_direction_slot(query):
    assert question_slot(query, 'location') == 'direction'


@pytest.mark.parametrize('arrow', ['↓', '⬇', '↑', '⬆'])
@pytest.mark.parametrize('value', ['STRAIGHT', 'BACK'])
def test_vertical_arrow_alone_cannot_authorize_travel(monkeypatch, arrow, value):
    request, values = outputs()
    values[1]['regions'][0]['content'] = f'EXIT {arrow}'
    values[2]['citations'][0].update(quote=f'EXIT {arrow}', value=value)
    result, _ = request_runtime(monkeypatch, 'nemotron-nano-vl-8b', values, request)
    assert result['policy'] is None
    assert result['output']['proposed_output']['status'] == 'uncertain'
    assert result['output']['proposed_action'] is None
    assert direction_values(arrow) == []


def test_literal_gate_also_does_not_treat_down_as_back():
    import re
    assert not re.search(DIRECTIONS['back'], 'EXIT ↓')
    assert direction_values('Go back') == ['back']
