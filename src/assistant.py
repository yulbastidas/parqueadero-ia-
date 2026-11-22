"""Respuestas por reglas sobre los estados observados."""

from src.occupancy import Occupancy


def answer_status(question: str, state: Occupancy | None) -> str:
    if state is None or state.total == 0:
        return "Todavía no hay resultados de ocupación."
    if state.pending:
        return f"Hay {len(state.available)} plazas libres confirmadas y {len(state.pending)} en confirmación."
    if "Cuáles" in question:
        return "Plazas libres: " + ", ".join(state.available) if state.available else "No hay plazas libres."
    if "ocupación" in question:
        return f"La ocupación es del {state.occupancy_percent:.1f} % en el último fotograma analizado."
    if "lleno" in question:
        return "El parqueadero está lleno." if not state.available else f"Quedan {len(state.available)} plazas libres."
    return f"Hay {len(state.available)} plazas libres en el último fotograma analizado."
