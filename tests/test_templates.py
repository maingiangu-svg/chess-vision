import chess
import cv2
import numpy as np
from vision.square_mapping import point_for_square
from vision.template_recognition import TemplateRecognizer, SYMBOL_TO_LABEL, LABELS

def painted_start(orientation="white"):
    image = np.zeros((800,800,3), np.uint8)
    board = chess.Board()
    for square in chess.SQUARES:
        name = chess.square_name(square)
        x,y = point_for_square(name, orientation=orientation)
        x,y = int(x-50),int(y-50)
        color = (130,185,130) if (x//100+y//100)%2 else (225,235,225)
        image[y:y+100,x:x+100] = color
        piece = board.piece_at(square)
        if piece:
            label = SYMBOL_TO_LABEL[piece.symbol()]
            glyph = f"{LABELS.index(label):02d}"
            ink = (245,245,245) if piece.color else (30,30,30)
            cv2.putText(image,glyph,(x+13,y+67),cv2.FONT_HERSHEY_SIMPLEX,1.3,ink,5)
    return image

def test_template_calibration_and_four_rotations():
    for orientation in ("white","black","white_left","white_right"):
        image = painted_start(orientation)
        model = TemplateRecognizer.calibrate(image,chess.Board(),orientation)
        found, confidence, errors = model.predict(image,orientation)
        assert len(found)==32 and not errors
        assert confidence > 0
