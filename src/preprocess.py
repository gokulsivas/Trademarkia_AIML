import re
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# DESIGN DECISIONS (justification lives here per task brief)
#
# 1. We clean the body only — headers are already parsed into CSV columns.
#    The subject line is prepended to the body because it is often the densest
#    semantic signal in short posts, and transformer attention benefits from
#    topic-anchoring at the start of the input sequence.
#
# 2. We do NOT lowercase, stem, or remove stopwords.
#    sentence-transformers operate on subword tokens and are trained on cased
#    text. Lowercasing destroys proper noun distinctions (NASA vs nasa),
#    stemming corrupts morphology the model was trained to interpret.
#
# 3. We keep technical jargon intact (SCSI, BIOS, X11, TCP/IP).
#    These are the primary discriminators between comp.* subcategories.
#    Stripping them would collapse semantically distinct clusters.
#
# 4. UUEncoded blocks are removed aggressively.
#    They produce hundreds of random ASCII tokens per document that form
#    a spurious "encoding" cluster with zero semantic meaning.
#
# 5. Four quote styles are handled (>, :, ->, }) — all confirmed present
#    in the raw CSV. Quoted text represents someone else's semantic context,
#    not the document's own. Including it pulls embeddings toward the
#    quoted article's topic rather than the reply's.
#
# 6. Documents under 50 characters after cleaning are flagged as short_doc.
#    They embed into unstable vector positions due to insufficient semantic
#    mass. They are retained in the index but excluded from cluster centroid
#    calculations.
# ---------------------------------------------------------------------------

# Compiled regex patterns — compiled once at module load for performance
_UUENCODE_BLOCK   = re.compile(r'begin \d{3} \S+.*?end', re.DOTALL)
_ATTRIBUTION_LINE = re.compile(
    r'^(In article\b.*?writes?:?|.*?<.*?>\s*writes?:?|.*wrote:)\s*$',
    re.MULTILINE | re.IGNORECASE
)
_QUOTE_LINE       = re.compile(r'^\s*(>+|:+|-{1,2}>|\}+).*$', re.MULTILINE)
_SEPARATOR_LINE   = re.compile(r'^[-=*]{3,}\s*$', re.MULTILINE)
_DELETION_MARKER  = re.compile(r'\[.*?(deleted?|snipped?|trimmed?|truncated?).*?\]', re.IGNORECASE)
_EMAIL_ADDRESS    = re.compile(r'\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b')
_URL              = re.compile(r'https?://\S+|www\.\S+')
_PHONE_NUMBER     = re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b')
_ASCII_BOX_LINE   = re.compile(r'^[|*]{1}.*[|*]{1}\s*$', re.MULTILINE)
_EXCESS_WHITESPACE = re.compile(r'\n{3,}')


def _remove_signature(text: str) -> str:
    """
    Remove signatures using the standard Usenet '-- ' delimiter.
    Also removes ASCII box signatures (lines bounded by | or *) 
    which are common in university/corporate accounts in this corpus.
    """
    # Standard delimiter: double-dash on its own line
    sig_match = re.search(r'\n--\s*\n', text)
    if sig_match:
        text = text[:sig_match.start()]

    # ASCII box signatures: remove lines that look like | content | or * content *
    text = _ASCII_BOX_LINE.sub('', text)
    return text


def clean_body(subject: str, body: str) -> str:
    """
    Full cleaning pipeline for a single document body.
    Returns the cleaned text with subject prepended.
    """
    text = str(body) if pd.notna(body) else ''

    # Stage 1: Remove UUEncoded binary blocks (confirmed present in corpus)
    text = _UUENCODE_BLOCK.sub('', text)

    # Stage 2: Remove signature blocks before quote removal
    # (some signatures contain > characters that would confuse quote stripping)
    text = _remove_signature(text)

    # Stage 3: Remove article attribution lines (e.g. "In article <...> X writes:")
    text = _ATTRIBUTION_LINE.sub('', text)

    # Stage 4: Remove all four quote styles confirmed in this corpus
    text = _QUOTE_LINE.sub('', text)

    # Stage 5: Remove ASCII separator lines
    text = _SEPARATOR_LINE.sub('', text)

    # Stage 6: Remove explicit deletion/truncation markers
    text = _DELETION_MARKER.sub('', text)

    # Stage 7: Remove emails, URLs, phone numbers
    # These are high-cardinality tokens with no semantic generalization
    text = _EMAIL_ADDRESS.sub('', text)
    text = _URL.sub('', text)
    text = _PHONE_NUMBER.sub('', text)

    # Stage 8: Collapse excess whitespace
    text = _EXCESS_WHITESPACE.sub('\n\n', text)
    text = text.strip()

    # Stage 9: Prepend subject as topic anchor
    # Subject is extracted separately since it is already in a CSV column.
    subject_str = str(subject).strip() if pd.notna(subject) else ''
    if subject_str and subject_str.lower() not in ('nan', '', 'none'):
        # Strip "Re:" prefixes — they add noise without semantic value
        subject_clean = re.sub(r'^(Re:\s*)+', '', subject_str, flags=re.IGNORECASE).strip()
        if subject_clean:
            text = f"Subject: {subject_clean}\n\n{text}"

    return text


def load_and_preprocess(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the newsgroups CSV and apply the full preprocessing pipeline.
    
    Returns a DataFrame with all original columns plus:
      - clean_text     : subject-prepended, fully cleaned body text
      - is_crossposted : True if the article appeared in multiple newsgroups
      - body_len_raw   : character count before cleaning
      - body_len_clean : character count after cleaning
      - short_doc      : True if clean_text < 50 chars (unstable embedding territory)
    """
    df = pd.read_csv(csv_path, dtype=str)

    print(f"[preprocess] Loaded {len(df)} documents from {csv_path}")

    # Detect cross-posted documents from the newsgroups column
    # These are natural fuzzy-cluster boundary cases and are flagged
    # for special treatment during cluster analysis
    df['is_crossposted'] = df['newsgroups'].str.contains(',', na=False)

    # Raw body length before cleaning
    df['body_len_raw'] = df['body'].fillna('').str.len()

    # Apply the cleaning pipeline to every document
    print("[preprocess] Cleaning documents...")
    df['clean_text'] = df.apply(
        lambda row: clean_body(row['subject'], row['body']),
        axis=1
    )

    # Post-cleaning length and short document flag
    df['body_len_clean'] = df['clean_text'].str.len()
    df['short_doc'] = df['body_len_clean'] < 50

    # Report cleaning statistics
    total       = len(df)
    short_count = df['short_doc'].sum()
    cross_count = df['is_crossposted'].sum()
    avg_raw     = df['body_len_raw'].mean()
    avg_clean   = df['body_len_clean'].mean()

    print(f"[preprocess] Done.")
    print(f"  Total documents   : {total}")
    print(f"  Cross-posted      : {cross_count} ({100*cross_count/total:.1f}%)")
    print(f"  Short docs (<50ch): {short_count} ({100*short_count/total:.1f}%)")
    print(f"  Avg length raw    : {avg_raw:.0f} chars")
    print(f"  Avg length clean  : {avg_clean:.0f} chars")

    return df


if __name__ == '__main__':
    # Quick smoke test — run with: python -m src.preprocess
    data_path = Path('data/20_newsgroups.csv')
    df = load_and_preprocess(data_path)
    print("\nSample cleaned document:")
    sample = df[~df['short_doc']].iloc[0]
    print(f"  Category : {sample['category']}")
    print(f"  Raw len  : {sample['body_len_raw']}")
    print(f"  Clean len: {sample['body_len_clean']}")
    print(f"  Text preview:\n{sample['clean_text'][:400]}")
