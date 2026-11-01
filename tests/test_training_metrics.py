import pytest
from training.data import SQUARES
from training.metrics import summarize


def test_exact_board_and_abstention_cannot_be_hidden():
    truth = {s: "empty" for s in SQUARES}
    observed = dict(truth)
    observed["a1"] = "uncertain"
    result = summarize([dict(truth=truth, predicted=observed, latency_ms=10, tags=["glare"])])
    assert result["square_accuracy"] == pytest.approx(63/64)
    assert result["full_board_exact_match"] == 0
    assert result["abstention_rate"] == pytest.approx(1/64)
    assert result["invalid_position_frequency"] == 1
    assert result["by_tag"]["glare"]["boards"] == 1
    with pytest.raises(ValueError):
        summarize([])
