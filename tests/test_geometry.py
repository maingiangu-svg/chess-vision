import numpy as np
import cv2
from vision.square_mapping import square_from_point, point_for_square
from vision.board_detection import order_corners, BoardGeometry, detect_board_corners
from tests.test_templates import painted_start

def test_orientation_mapping():
    assert square_from_point(50, 50) == "a8"
    assert square_from_point(750, 750, orientation="black") == "a8"
    assert point_for_square("e4") == (450.0, 450.0)
    for orientation in ("white", "black", "white_left", "white_right"):
        assert square_from_point(*point_for_square("a1", orientation=orientation), orientation=orientation) == "a1"
        assert square_from_point(*point_for_square("h8", orientation=orientation), orientation=orientation) == "h8"
    assert square_from_point(50, 50, orientation="white_left") == "a1"

def test_corner_order_and_homography():
    corners = order_corners(np.array([[100,100],[900,120],[880,900],[120,880]]))
    assert np.allclose(corners[0], [100,100])
    h, inv = BoardGeometry(corners).homography()
    assert np.allclose(h @ inv, np.eye(3), atol=1e-4)

def test_board_detection_rotated_and_perspective():
    destination = np.array([[120,130],[850,100],[880,870],[100,830]], np.float32)
    source = np.array([[0,0],[799,0],[799,799],[0,799]], np.float32)
    transform = cv2.getPerspectiveTransform(source, destination)
    for rotation in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
        board = painted_start()
        if rotation is not None: board = cv2.rotate(board, rotation)
        frame = cv2.warpPerspective(board, transform, (1000, 1000))
        found = detect_board_corners(frame)
        assert found is not None
        assert np.linalg.norm(found.corners.mean(axis=0)-destination.mean(axis=0)) < 60
