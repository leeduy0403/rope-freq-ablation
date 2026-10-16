from src.tasks.fwe import FWE
from src.tasks.niah import NIAH
from src.tasks.vt import VT


def get_task(name, task_params=None):
    p = (task_params or {}).get(name, {})
    if name == "s_niah":
        return NIAH(n_distractors=0)
    if name == "mk_niah":
        return NIAH(n_distractors=p.get("n_distractors", 3))
    if name == "vt":
        return VT(**p)
    if name == "fwe":
        return FWE(**p)
    raise KeyError(name)
