"""Tests for SemanticWord 64-bit encoding."""
import pytest
from src.semantics.semantic_word import SemanticWord, WordType, IntentType, ChannelType


def test_encode_decode_roundtrip():
    w = SemanticWord(type=1, intent=2, channel=3, priority=4,
                     confidence=12345, payload_ref=54321)
    assert SemanticWord.decode(w.encode()) == w


def test_all_fields_max():
    w = SemanticWord(type=255, intent=255, channel=255, priority=255,
                     confidence=65535, payload_ref=65535)
    assert SemanticWord.decode(w.encode()) == w


def test_all_fields_zero():
    w = SemanticWord(type=0, intent=0, channel=0, priority=0,
                     confidence=0, payload_ref=0)
    assert w.encode() == 0
    assert SemanticWord.decode(0) == w


def test_make_factory():
    w = SemanticWord.make(
        type=WordType.MODEL,
        intent=IntentType.PLAN,
        channel=ChannelType.USER,
        priority=200,
        confidence=0.5,
        payload_ref=42,
    )
    assert w.type == int(WordType.MODEL)
    assert w.intent == int(IntentType.PLAN)
    assert w.channel == int(ChannelType.USER)
    assert w.priority == 200
    assert abs(w.confidence_f - 0.5) < 0.01
    assert w.payload_ref == 42


def test_confidence_float_roundtrip():
    for conf in [0.0, 0.25, 0.5, 0.75, 1.0]:
        w = SemanticWord.make(confidence=conf)
        assert abs(w.confidence_f - conf) < 0.001


def test_field_isolation():
    """Each field occupies independent bits."""
    for field_name, shift, mask in [
        ("type",        56, 0xFF),
        ("intent",      48, 0xFF),
        ("channel",     40, 0xFF),
        ("priority",    32, 0xFF),
        ("confidence",  16, 0xFFFF),
        ("payload_ref",  0, 0xFFFF),
    ]:
        w = SemanticWord(type=0, intent=0, channel=0, priority=0,
                         confidence=0, payload_ref=0)
        setattr(w, field_name, mask)
        encoded = w.encode()
        extracted = (encoded >> shift) & mask
        assert extracted == mask, f"{field_name} at shift={shift}"


def test_repr_readable():
    w = SemanticWord.make(type=WordType.RESULT, intent=IntentType.EMIT)
    assert "RESULT" in repr(w)
    assert "EMIT" in repr(w)
