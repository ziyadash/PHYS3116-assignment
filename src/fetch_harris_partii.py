"""
Regenerate data/HarrisPartII.csv from the published Harris catalogue.

Part II was missing from the course-supplied files. It is the only part carrying [Fe/H]
for the whole catalogue rather than the 55 clusters VandenBerg covers.

This exists so the parsed file is reproducible rather than something that appeared in the
repo without explanation. Run with:  python src/fetch_harris_partii.py

See data/SOURCES.md for provenance.
"""

import csv
import re
import urllib.request

from config import DATA_FILEPATHS

CATALOGUE_URL = 'https://physics.mcmaster.ca/~harris/mwgc.dat'

# The file is fixed-width. The identifier occupies the first 12 characters and [Fe/H] the
# next six. Some clusters have no published metallicity and leave that field blank.
ID_COLUMNS = slice(0, 12)
FEH_COLUMNS = slice(12, 18)


def parse_part_two(text):
    """Pull (identifier, [Fe/H]) out of the Part II block."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if 'Part II:  Metallicity' in line)
    end = next(i for i, line in enumerate(lines) if 'Part III' in line)

    rows = []
    for line in lines[start:end]:
        if not line.startswith(' ') or len(line) < 20:
            continue
        identifier = line[ID_COLUMNS].strip()
        if not identifier or identifier.startswith(('ID', 'Key', '(', '-')):
            continue
        metallicity = line[FEH_COLUMNS].strip()
        if not re.fullmatch(r'-?\d+\.\d+', metallicity or ''):
            metallicity = ''
        rows.append((identifier, metallicity))
    return rows


def main():
    with urllib.request.urlopen(CATALOGUE_URL, timeout=60) as response:
        text = response.read().decode('utf-8', errors='replace')

    rows = parse_part_two(text)
    path = DATA_FILEPATHS['HarrisPartII']
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['ID', 'FeH'])
        writer.writerows(rows)

    with_metallicity = sum(1 for _, feh in rows if feh)
    print(f'wrote {path.name}: {len(rows)} clusters, {with_metallicity} with [Fe/H]')
    print('missing:', [i for i, f in rows if not f])


if __name__ == '__main__':
    main()
