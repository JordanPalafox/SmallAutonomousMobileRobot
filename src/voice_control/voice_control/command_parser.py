"""Grammar-based command parser for the AMR voice interface.

Maps recognised word sequences to structured robot commands.
Vocabulary matches the 10 words trained in the HMM system.

Vocabulary: one, two, racks, rollers, map, stop, forward, left, right, continue
"""

from __future__ import annotations

VOCABULARY: list[str] = [
    'one', 'two', 'racks', 'rollers', 'map',
    'stop', 'forward', 'left', 'right', 'continue',
]

# ── Grammar rules ──────────────────────────────────────────────────────────────
# Each rule: (pattern_fn, result_fn)
# pattern_fn receives the full word buffer (list[str])
# result_fn  receives the same buffer and returns a command dict

_RULES: list[tuple] = [

    # ── Two-word location commands ──────────────────────────────────────────
    # "racks one"   → go to rack_1
    (
        lambda w: len(w) >= 2 and w[-2] == 'racks' and w[-1] == 'one',
        lambda w: {'cmd': 'goto', 'location': 'rack_1'},
    ),
    # "racks two"   → go to rack_2
    (
        lambda w: len(w) >= 2 and w[-2] == 'racks' and w[-1] == 'two',
        lambda w: {'cmd': 'goto', 'location': 'rack_2'},
    ),
    # "rollers one" → go to roller_1
    (
        lambda w: len(w) >= 2 and w[-2] == 'rollers' and w[-1] == 'one',
        lambda w: {'cmd': 'goto', 'location': 'roller_1'},
    ),
    # "rollers two" → go to roller_2
    (
        lambda w: len(w) >= 2 and w[-2] == 'rollers' and w[-1] == 'two',
        lambda w: {'cmd': 'goto', 'location': 'roller_2'},
    ),

    # ── Single-word movement commands ────────────────────────────────────────
    (lambda w: w[-1:] == ['stop'],     lambda w: {'cmd': 'stop'}),
    (lambda w: w[-1:] == ['continue'], lambda w: {'cmd': 'continue'}),
    (lambda w: w[-1:] == ['forward'],  lambda w: {'cmd': 'move', 'direction': 'forward'}),
    (lambda w: w[-1:] == ['left'],     lambda w: {'cmd': 'turn', 'direction': 'left'}),
    (lambda w: w[-1:] == ['right'],    lambda w: {'cmd': 'turn', 'direction': 'right'}),

    # ── Single-word location commands ────────────────────────────────────────
    # Standalone "racks" / "rollers" without a number → list available locations
    (lambda w: w[-1:] == ['racks'],   lambda w: {'cmd': 'goto', 'location': 'racks'}),
    (lambda w: w[-1:] == ['rollers'], lambda w: {'cmd': 'goto', 'location': 'rollers'}),

    # ── Map command ──────────────────────────────────────────────────────────
    (lambda w: w[-1:] == ['map'],     lambda w: {'cmd': 'show_map'}),

    # ── Numeric level / selection commands ──────────────────────────────────
    (lambda w: w[-1:] == ['one'],     lambda w: {'cmd': 'select', 'value': 1}),
    (lambda w: w[-1:] == ['two'],     lambda w: {'cmd': 'select', 'value': 2}),
]


def parse_command(word_sequence: list[str]) -> dict | None:
    """Match a word sequence to a structured command.

    Evaluates rules in order; the first match wins.
    Two-word rules are checked before single-word rules so that
    "racks one" is not consumed as two separate "racks"/"one" commands.

    Returns a dict or None if no rule matches.
    """
    if not word_sequence:
        return None
    for pattern_fn, result_fn in _RULES:
        if pattern_fn(word_sequence):
            return result_fn(word_sequence)
    return None
