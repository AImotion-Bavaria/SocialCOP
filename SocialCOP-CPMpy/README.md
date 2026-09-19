# SocialCOP for CPMpy

This folder is the CPMpy counterpart of the MiniZinc implementation in the
repository root. It is independent: neither the `minizinc` Python package nor a
MiniZinc installation is required.

## Setup

```powershell
cd SocialCOP-CPMpy
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
pytest
```

Open `SocialCOP.code-workspace` from the repository root to see the MiniZinc
and CPMpy implementations as separate workspace folders.

The folder is deliberately named `SocialCOP-CPMpy` rather than `cpmpy`, because
a project folder named `cpmpy` would shadow the installed Python package when
commands are run from the repository root.

## Layout

- `src/socialcop_cpmpy/social.py`: reusable fairness and welfare constraints
- `src/socialcop_cpmpy/runners.py`: solver entry points
- `src/models/`: CPMpy model categories mirroring `../src/models/`
- `src/models/table_assignment/model.py`: runnable native CPMpy example

The directory structure is ready for models to be ported incrementally. A
MiniZinc `.mzn` file cannot be executed by CPMpy; each model's variables and
constraints must be expressed in Python.
