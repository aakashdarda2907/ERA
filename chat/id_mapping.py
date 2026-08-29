import csv
from functools import lru_cache
from pathlib import Path


MAPPING_FILE = Path(__file__).resolve().parent.parent / "data" / "IDS_mapping.csv"


def _normalise_code(value):
    """Convert ID values to a consistent string representation."""
    if value is None:
        return ""

    value = str(value).strip()

    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass

    return value


@lru_cache(maxsize=1)
def load_id_mappings():
    """Load all ID descriptions from IDS_mapping.csv."""

    mappings = {}
    current_column = None

    with MAPPING_FILE.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)

        for row in reader:
            if not row:
                continue

            first = row[0].strip()
            second = row[1].strip() if len(row) > 1 else ""

            if first in {
                "admission_type_id",
                "discharge_disposition_id",
                "admission_source_id",
            } and second == "description":
                current_column = first
                mappings[current_column] = {}
                continue

            if current_column is None:
                continue

            if not first or len(row) < 2:
                continue

            description = row[1].strip()
            mappings[current_column][_normalise_code(first)] = description

    return mappings


def get_id_description(column_name, code):
    """Return the readable description for an ID code."""

    mappings = load_id_mappings()
    column_mapping = mappings.get(column_name, {})

    return column_mapping.get(_normalise_code(code))