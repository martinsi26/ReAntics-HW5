#!/bin/bash

# Usage: ./split_random.sh input.csv
# Output: input_part1.csv ... input_part8.csv

INPUT="$1"
NUM_SPLITS=8

if [ -z "$INPUT" ]; then
  echo "Usage: $0 <input.csv>"
  exit 1
fi

HEADER=$(head -n 1 "$INPUT")
TMPFILE=$(mktemp)

# Shuffle the CSV excluding the header
tail -n +2 "$INPUT" | shuf > "$TMPFILE"

# Count total lines (excluding header)
TOTAL_LINES=$(wc -l < "$TMPFILE")
LINES_PER_SPLIT=$(( (TOTAL_LINES + NUM_SPLITS - 1) / NUM_SPLITS ))

# Split into 8 roughly equal random parts
split -l "$LINES_PER_SPLIT" "$TMPFILE" "${INPUT%.csv}_part"

# Add header to each part and rename properly
i=1
for f in ${INPUT%.csv}_part*; do
  NEWFILE="${INPUT%.csv}_part${i}.csv"
  { echo "$HEADER"; cat "$f"; } > "$NEWFILE"
  rm "$f"
  ((i++))
done

rm "$TMPFILE"

echo "✅ Split complete: created ${NUM_SPLITS} CSV files."
