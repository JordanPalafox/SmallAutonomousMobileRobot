"""Grammar-based command parser for the AMR voice interface.

Maps recognised word sequences to structured robot commands.
Vocabulary matches the 10 words trained in the HMM system.

Vocabulary: one, two, racks, rollers, stop, forward, left, right, continue

Command categories (used by voice_node to route each result):
  mission  → serialised as JSON to /mission
  control  → raw word to /voice_command  (state-machine pause/resume)
  motion   → Twist published to /cmd_vel_in while active
"""

from __future__ import annotations

VOCABULARY: list[str] = [
    'one', 'two', 'racks', 'rollers',
    'stop', 'forward', 'left', 'right', 'continue',
]

_RULES: list[tuple] = [

    # ── Mission 1: busca todos los rollers, recoge el pallet y lo entrega al
    #    camión que indica el QR  (ROLLER_TO_TRUCK)
    (
        lambda w: w[-1:] == ['one'],
        lambda w: {'category': 'mission', 'payload': {
            'type': 'ROLLER_TO_TRUCK', 'id': 'voice_1', 'source': {},
        }},
    ),

    # ── Mission 2: busca todos los racks, recoge y entrega (RACK_TO_TRUCK)
    (
        lambda w: w[-1:] == ['two'],
        lambda w: {'category': 'mission', 'payload': {
            'type': 'RACK_TO_TRUCK', 'id': 'voice_2', 'source': {},
        }},
    ),

    # ── Búsqueda de racks (sin pick ni entrega)
    (
        lambda w: w[-1:] == ['racks'],
        lambda w: {'category': 'mission', 'payload': {
            'type': 'SEARCH_RACKS', 'id': 'voice_racks', 'source': {},
        }},
    ),

    # ── Búsqueda de rollers (sin pick ni entrega)
    (
        lambda w: w[-1:] == ['rollers'],
        lambda w: {'category': 'mission', 'payload': {
            'type': 'SEARCH_ROLLERS', 'id': 'voice_rollers', 'source': {},
        }},
    ),

    # ── Control: detiene la misión y el movimiento
    (lambda w: w[-1:] == ['stop'],     lambda w: {'category': 'control', 'word': 'stop'}),

    # ── Control: reanuda la misión pausada
    (lambda w: w[-1:] == ['continue'], lambda w: {'category': 'control', 'word': 'resume'}),

    # ── Movimiento manual (continuo hasta recibir 'stop')
    (lambda w: w[-1:] == ['forward'],  lambda w: {'category': 'motion', 'direction': 'forward'}),
    (lambda w: w[-1:] == ['left'],     lambda w: {'category': 'motion', 'direction': 'left'}),
    (lambda w: w[-1:] == ['right'],    lambda w: {'category': 'motion', 'direction': 'right'}),
]


def parse_command(word_sequence: list[str]) -> dict | None:
    """Match the word buffer to a structured command (first rule wins)."""
    if not word_sequence:
        return None
    for pattern_fn, result_fn in _RULES:
        if pattern_fn(word_sequence):
            return result_fn(word_sequence)
    return None
