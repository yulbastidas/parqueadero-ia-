import pytest

from src.detector import Detection
from src.occupancy import OccupancyMonitor, validate_spaces


def test_stabilizes_changes_and_assigns_one_space_per_detection():
    spaces = validate_spaces({"spaces": [
        {"id": "A", "points": [[0, 0], [.4, 0], [.4, .5], [0, .5]]},
        {"id": "B", "points": [[.4, 0], [.8, 0], [.8, .5], [.4, .5]]},
    ]})
    monitor = OccupancyMonitor(spaces, confirmation=2)
    vehicle = Detection(3, "auto", .9, (20, 0, 50, 50))
    assert monitor.update((100, 100, 3), (vehicle,)).pending == ("A", "B")
    result = monitor.update((100, 100, 3), (vehicle,))
    assert (result.occupied, result.available, result.occupancy_percent) == (("A",), ("B",), 50)
    assert monitor.update((100, 100, 3), ()).occupied == ("A",)
    assert monitor.update((100, 100, 3), ()).occupancy_percent == 0


def test_empty_or_invalid_configuration():
    assert OccupancyMonitor({}).update((100, 100, 3), ()).occupancy_percent is None
    with pytest.raises(ValueError, match="inválido"):
        validate_spaces({"spaces": [{"id": "A", "points": [[0, 0], [1, 1], [1, 0], [0, 1]]}]})
