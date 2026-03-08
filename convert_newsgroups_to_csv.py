"""
convert_newsgroups_to_csv.py
-----------------------------
Converts 20 Newsgroups .tar.gz archives into clean CSV files.

Usage:
    python convert_newsgroups_to_csv.py

Requirements:
    pip install pandas

Input files expected in the same directory as this script:
    - mini_newsgroups_tar.gz
    - 20_newsgroups_tar.gz

Output:
    - mini_newsgroups.csv
    - 20_newsgroups.csv
"""

import tarfile
import re
import pandas as pd
from pathlib import Path


# ── CONFIG ─────────────────────────────────────────────────────────────────────
ARCHIVES = {
    r"C:\Users\gokul\Downloads\mini_newsgroups.tar.gz": "mini_newsgroups.csv",
    r"C:\Users\gokul\Downloads\20_newsgroups.tar.gz":   "20_newsgroups.csv",
}

# Header fields we want to extract from each post.
# These are standard USENET/email headers present in most posts.
HEADER_FIELDS = ["from", "subject", "date", "organization", "newsgroups", "lines"]
# ───────────────────────────────────────────────────────────────────────────────


def parse_post(raw_text: str) -> dict:
    """
    Parse a single newsgroup post (raw text) into a structured dict.

    Each post has an email-style header block at the top, followed by
    a blank line, then the body. We split on that blank line and parse
    each header field we care about. Everything after the blank line
    is the body.

    Noisy content we deliberately discard:
    - 'Path:' routing headers  (server hops, irrelevant to content)
    - 'Xref:' cross-post references
    - 'Message-ID:' and 'References:' (technical, not semantic)
    - Quoted reply lines starting with '>' (duplicate content from
      prior messages — including them would inflate similarity scores
      between posts that merely share a reply chain, not a topic)
    - Lines that are purely whitespace or email signatures ('-- ')
    """

    # Split header block from body on the first blank line
    parts = re.split(r'\n\n', raw_text, maxsplit=1)
    header_block = parts[0] if len(parts) > 0 else ""
    body_raw     = parts[1] if len(parts) > 1 else ""

    # ── Parse headers ──────────────────────────────────────────────────────────
    headers = {}
    current_key = None

    for line in header_block.splitlines():
        # A header line looks like "Key: value"
        match = re.match(r'^([A-Za-z\-]+):\s*(.*)', line)
        if match:
            current_key = match.group(1).lower()
            headers[current_key] = match.group(2).strip()
        elif current_key and line.startswith((' ', '\t')):
            # Continuation of a multi-line header value
            headers[current_key] += ' ' + line.strip()

    # ── Clean body ─────────────────────────────────────────────────────────────
    cleaned_lines = []
    in_signature = False

    for line in body_raw.splitlines():
        # Email signature delimiter — everything after this is noise
        if line.strip() == '--':
            in_signature = True
            continue
        if in_signature:
            continue

        # Skip quoted reply lines (start with '>' or '|')
        if re.match(r'^\s*[>|]', line):
            continue

        # Skip lines that are just whitespace
        if not line.strip():
            # Keep a single blank line for paragraph spacing, skip multiples
            if cleaned_lines and cleaned_lines[-1] != '':
                cleaned_lines.append('')
            continue

        cleaned_lines.append(line.strip())

    body_clean = '\n'.join(cleaned_lines).strip()

    # ── Build output record ────────────────────────────────────────────────────
    record = {field: headers.get(field, "") for field in HEADER_FIELDS}
    record["body"] = body_clean
    record["body_length"] = len(body_clean.split())  # word count, useful later

    return record


def extract_category(member_path: str) -> str:
    """
    Derive the newsgroup category from the tar member's path.

    Paths look like:  mini_newsgroups/sci.electronics/54165
    We want:          sci.electronics
    """
    parts = Path(member_path).parts
    # parts[0] = archive root dir, parts[1] = category, parts[2] = file id
    if len(parts) >= 3:
        return parts[1]
    elif len(parts) == 2:
        return parts[0]
    return "unknown"


def process_archive(archive_path: str, output_csv: str) -> None:
    """
    Open a .tar.gz archive, iterate over every post file,
    parse it, and write the results to a CSV.
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        print(f"  [SKIP] File not found: {archive_path}")
        return

    records = []

    print(f"\nProcessing: {archive_path.name}")

    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        total = len([m for m in members if m.isfile()])
        print(f"  Found {total} files in archive")

        for i, member in enumerate(members):
            # Skip directories and any hidden/system files
            if not member.isfile():
                continue
            if Path(member.name).name.startswith('.'):
                continue

            # Progress indicator every 1000 files
            if i > 0 and i % 1000 == 0:
                print(f"  ... processed {i} files")

            try:
                f = tar.extractfile(member)
                if f is None:
                    continue

                # Newsgroup posts are latin-1 encoded (old USENET era)
                # We fall back to replacing undecodable bytes rather than crashing
                raw_text = f.read().decode("latin-1", errors="replace")

            except Exception as e:
                print(f"  [WARN] Could not read {member.name}: {e}")
                continue

            record = parse_post(raw_text)
            record["file_id"]  = Path(member.name).name   # numeric post ID
            record["category"] = extract_category(member.name)

            records.append(record)

    if not records:
        print("  [ERROR] No records found. Check archive structure.")
        return

    # ── Build DataFrame & reorder columns ─────────────────────────────────────
    df = pd.DataFrame(records)

    # Reorder so the most useful columns come first
    col_order = ["file_id", "category", "subject", "from", "date",
                 "organization", "newsgroups", "lines", "body_length", "body"]
    # Only keep columns that actually exist
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    # ── Basic stats ────────────────────────────────────────────────────────────
    print(f"\n  ── Summary ──────────────────────────────")
    print(f"  Total posts      : {len(df)}")
    print(f"  Categories       : {df['category'].nunique()}")
    print(f"  Posts per category:\n{df['category'].value_counts().to_string()}")
    print(f"  Avg body length  : {df['body_length'].mean():.0f} words")
    print(f"  Empty bodies     : {(df['body'] == '').sum()}")

    # ── Save ───────────────────────────────────────────────────────────────────
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n  Saved → {output_csv}")
    print(f"  ─────────────────────────────────────────")


def main():
    print("=" * 50)
    print("  Newsgroups → CSV Converter")
    print("=" * 50)

    for archive_name, output_csv in ARCHIVES.items():
        process_archive(archive_name, output_csv)

    print("\nDone! Open the CSV files in Excel or load with pandas:")
    print("  import pandas as pd")
    print("  df = pd.read_csv('mini_newsgroups.csv')")
    print("  df.head()")


if __name__ == "__main__":
    main()