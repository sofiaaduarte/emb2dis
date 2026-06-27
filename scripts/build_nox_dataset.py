from pathlib import Path
import pandas as pd

# ----------------------------------- Paths ---------------------------------- #
INPUT_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/nox")

TRAIN_CSV = INPUT_DIR / "train.csv"
DEV_CSV = INPUT_DIR / "dev.csv"

FASTA_FILE = INPUT_DIR / "seqs.fasta"

OUT_TRAIN_CSV = OUTPUT_DIR / "train.csv"
OUT_DEV_CSV = OUTPUT_DIR / "dev.csv"

# --------------------------------- Functions -------------------------------- #
def read_fasta_lengths(fasta_path):
    """
    Read a FASTA file and return a dictionary with sequence lengths.
    """
    lengths = {}

    current_id = None
    current_seq = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if current_id is not None:
                    lengths[current_id] = len("".join(current_seq))

                # Use the first token after ">" as the sequence ID
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

        if current_id is not None:
            lengths[current_id] = len("".join(current_seq))

    return lengths


def merge_intervals(intervals):
    """
    Merge overlapping or consecutive intervals.

    Intervals are assumed to be 1-based and inclusive:
        start, end
    """
    if not intervals:
        return []

    intervals = sorted(intervals)
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]

        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def get_complement_intervals(disordered_intervals, sequence_length):
    """
    Generate structured intervals as the complement of disordered intervals.

    That is, structured residues are all residues that are not annotated
    as disordered.
    """
    structured_intervals = []
    current_start = 1

    for start, end in disordered_intervals:
        if current_start < start:
            structured_intervals.append((current_start, start - 1))

        current_start = end + 1

    if current_start <= sequence_length:
        structured_intervals.append((current_start, sequence_length))

    return structured_intervals


def rebuild_annotations(csv_path, fasta_lengths, output_path):
    """
    Rebuild the annotation file using the new labeling paradigm.

    The original structured annotations are ignored.
    The new structured annotations are defined as the complement of the
    disordered annotations over the full sequence length.
    """
    df = pd.read_csv(csv_path)

    required_columns = {"acc", "start", "end", "label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in {csv_path}: {missing_columns}"
        )

    new_rows = []

    for acc, group in df.groupby("acc", sort=True):
        if acc not in fasta_lengths:
            raise ValueError(
                f"Sequence {acc} was found in {csv_path}, "
                f"but not in the FASTA file."
            )

        sequence_length = fasta_lengths[acc]

        disordered_df = group[group["label"] == "disordered"]

        disordered_intervals = list(
            disordered_df[["start", "end"]].itertuples(index=False, name=None)
        )

        disordered_intervals = merge_intervals(disordered_intervals)

        for start, end in disordered_intervals:
            if start < 1 or end > sequence_length or start > end:
                raise ValueError(
                    f"Invalid interval for {acc}: {start}-{end}. "
                    f"Sequence length: {sequence_length}"
                )

        # Keep disordered regions as positive annotations
        for start, end in disordered_intervals:
            new_rows.append({
                "acc": acc,
                "start": start,
                "end": end,
                "label": "disordered"
            })

        # Rebuild structured regions as everything that is not disordered
        structured_intervals = get_complement_intervals(
            disordered_intervals,
            sequence_length
        )

        for start, end in structured_intervals:
            new_rows.append({
                "acc": acc,
                "start": start,
                "end": end,
                "label": "structured"
            })

    new_df = pd.DataFrame(new_rows)

    new_df = new_df.sort_values(
        by=["acc", "start", "end"]
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Number of annotations: {len(new_df)}")


# =========================
# Run
# =========================

fasta_lengths = read_fasta_lengths(FASTA_FILE)

rebuild_annotations(TRAIN_CSV, fasta_lengths, OUT_TRAIN_CSV)
rebuild_annotations(DEV_CSV, fasta_lengths, OUT_DEV_CSV)