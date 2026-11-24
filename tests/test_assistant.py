from src.assistant import answer_status
from src.occupancy import Occupancy


def test_answers_use_same_counts_ids_and_percentage_as_dashboard():
    state = Occupancy(("A1", "A2"), ("B1", "B2", "B3"), (), 5, 40.0)
    assert answer_status("libres", state) == "Hay 3 plazas libres confirmadas."
    assert answer_status("ubicacion", state) == "Plazas libres: B1, B2, B3"
    assert answer_status("porcentaje", state) == "La ocupación es del 40.0 %."
    assert answer_status("lleno", state) == "Quedan 3 plazas libres."


def test_answers_handle_pending_full_missing_and_out_of_scope():
    pending = Occupancy(("A1",), (), ("A2",), 2, None)
    full = Occupancy(("A1", "A2"), (), (), 2, 100.0)
    assert "pendiente de confirmación" in answer_status("porcentaje", pending)
    assert "A2" in answer_status("ubicacion", pending)
    assert "Aún no se puede confirmar" in answer_status("lleno", pending)
    assert answer_status("lleno", full) == "El parqueadero está lleno."
    assert "Todavía no hay" in answer_status("libres", None)
    assert "Puedo responder" in answer_status("fuera", full)
