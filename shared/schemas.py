from dataclasses import dataclass, field
from typing import Any

PIECE_CLASSES = ("white_pawn", "white_knight", "white_bishop", "white_rook", "white_queen", "white_king", "black_pawn", "black_knight", "black_bishop", "black_rook", "black_queen", "black_king")

@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    square: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class HudMove:
    uci: str
    from_square: str
    to_square: str
    score_cp: int | None
    pv: tuple[str, ...] = ()
