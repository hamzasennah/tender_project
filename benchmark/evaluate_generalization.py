import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django  # noqa: E402

django.setup()

from benchmark.generalization_evaluation import (  # noqa: E402
    REPORT_PATH,
    printable_summary,
    run_evaluation,
)


def main():
    report = run_evaluation(REPORT_PATH)
    print(printable_summary(report))
    print(f"\nReport written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
