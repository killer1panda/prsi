#!/bin/bash
# Production Data Pipeline Runner for PRSI Doom Index
# Unifies labeled datasets and prepares for semi-supervised training

set -e

echo "=========================================="
echo "PRSI Doom Index - Data Pipeline"
echo "=========================================="

# Configuration
DATA_ROOT="${DATA_ROOT:-/home/vivek.120542}"
OUTPUT_DIR="${OUTPUT_DIR:-data/unified}"
PUSHSHIFT_OUTPUT="${PUSHSHIFT_OUTPUT:-data/pushshift}"
TARGET_SAMPLES="${TARGET_SAMPLES:-10000}"

echo "Data Root: $DATA_ROOT"
echo "Output Directory: $OUTPUT_DIR"
echo "Target Samples per Source: $TARGET_SAMPLES"
echo ""

# Step 1: Unify labeled datasets
echo "[Step 1/3] Unifying labeled datasets..."
python -m src.data.unify_datasets \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --target-samples "$TARGET_SAMPLES"

# Step 2: Extract unlabeled texts from Pushshift
echo ""
echo "[Step 2/3] Preparing unlabeled Pushshift data..."
if [ -f "$DATA_ROOT/RC_2023-01.zst" ]; then
    echo "Found Pushshift archive, processing..."
    python -m src.data.pushshift_ingestion \
        --input "$DATA_ROOT/RC_2023-01.zst" \
        --output "$PUSHSHIFT_OUTPUT/reddit_unlabeled.parquet" \
        --n-workers 8 \
        --mode unlabeled_only
else
    echo "Warning: Pushshift archive not found at $DATA_ROOT/RC_2023-01.zst"
    echo "Skipping unlabeled data preparation"
fi

# Step 3: Generate statistics
echo ""
echo "[Step 3/3] Generating dataset statistics..."
python << PYTHON_SCRIPT
import pandas as pd
import json
from pathlib import Path

output_dir = Path("$OUTPUT_DIR")
stats = {}

for split in ['train', 'validation', 'test']:
    path = output_dir / f"{split}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        stats[split] = {
            'total_samples': len(df),
            'positive_class': int((df['label'] == 1).sum()),
            'negative_class': int((df['label'] == 0).sum()),
            'class_balance': float((df['label'] == 1).mean()),
            'sources': df['source'].unique().tolist(),
            'avg_text_length': float(df['text'].str.len().mean())
        }

with open(output_dir / 'dataset_report.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("\nDataset Statistics:")
print("=" * 60)
for split, s in stats.items():
    print(f"\n{split.upper()}:")
    print(f"  Total: {s['total_samples']:,}")
    print(f"  Positive: {s['positive_class']:,} ({s['class_balance']*100:.1f}%)")
    print(f"  Negative: {s['negative_class']:,}")
    print(f"  Sources: {', '.join(s['sources'])}")
    print(f"  Avg Text Length: {s['avg_text_length']:.1f}")

PYTHON_SCRIPT

echo ""
echo "=========================================="
echo "✅ Data pipeline completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review dataset statistics in $OUTPUT_DIR/dataset_report.json"
echo "  2. Run semi-supervised training:"
echo "     python -m src.training.semi_supervised_trainer \\"
echo "       --strategy self_training \\"
echo "       --labeled-data $OUTPUT_DIR/train.parquet \\"
echo "       --val-data $OUTPUT_DIR/validation.parquet \\"
echo "       --unlabeled-data $PUSHSHIFT_OUTPUT/reddit_unlabeled.parquet"
echo ""
