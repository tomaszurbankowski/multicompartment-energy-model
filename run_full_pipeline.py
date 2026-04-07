from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_SCRIPT = BASE_DIR / 'virtual_phenotypes_energy_model.py'
FIGURE_SCRIPT = BASE_DIR / 'generate_publication_figures.py'


def run(script: Path, cwd: Path) -> None:
    if not script.exists():
        raise FileNotFoundError(f'Script not found: {script}')
    result = subprocess.run([sys.executable, str(script)], cwd=str(cwd))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    run(MODEL_SCRIPT, BASE_DIR)
    run(FIGURE_SCRIPT, BASE_DIR)
    print('Full pipeline completed successfully.')


if __name__ == '__main__':
    main()
