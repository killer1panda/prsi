#!/usr/bin/env python3
"""WebDataset + Arrow Converter for H100 GPU Saturation.

Converts processed Reddit Parquet/CSV into WebDataset tar shards or
PyArrow IPC format for zero-copy, high-throughput GPU data loading.

Why not CSV/Pandas for H100 training?
- CSV parsing is single-threaded and CPU-bound
- Pandas DataFrames fragment memory and cause OOM
- WebDataset streams tar shards with multiple workers, saturating GPU
- Arrow memory maps enable true zero-copy from disk to GPU

Features:
- Shard generation with deterministic splitting (train/val/test)
- Tokenization pre-caching with HuggingFace Tokenizer
- Compression options (gz, zstd) for I/O vs CPU tradeoff
- Automatic sequence padding and attention mask generation
- Metadata JSON sidecars for label distribution tracking
- Integration with PyTorch IterableDataset

Usage:
    python -m src.data.webdataset_converter \
        --input data/reddit_processed.parquet \
        --output data/shards/ \
        --tokenizer distilbert-base-uncased \
        --max-len 256 \
        --shard-size 10000 \
        --compression zstd
"""

import argparse
import io
import json
import logging
import os
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    import webdataset as wds
    WEBDATASET_AVAILABLE = True
except ImportError:
    WEBDATASET_AVAILABLE = False
    logger.warning("webdataset not installed. Install: pip install webdataset")

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ShardWriter:
    """Write training examples into WebDataset tar shards."""

    def __init__(
        self,
        output_dir: str,
        shard_size: int = 10000,
        maxsize: int = 1e9,
        compress: bool = True,
        compression: str = "gz",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.maxsize = maxsize
        self.compress = compress
        self.compression = compression
        self._shard_idx = 0
        self._current_count = 0
        self._tar = None
        self._open_new_shard()
        self._meta = defaultdict(int)

    def _open_new_shard(self) -> None:
        """Rotate to a new tar shard file."""
        if self._tar:
            self._tar.close()
        suffix = f"{self._shard_idx:06d}"
        ext = f".tar.{self.compression}" if self.compress else ".tar"
        path = self.output_dir / f"shard_{suffix}{ext}"
        mode = f"w:{self.compression}" if self.compress else "w"
        self._tar = tarfile.open(path, mode)
        self._current_count = 0
        self._shard_idx += 1
        logger.info(f"Opened new shard: {path}")

    def write(self, key: str, data: Dict[str, bytes]) -> None:
        """Write an example to the current shard.

        Args:
            key: Unique example identifier
            data: Dict of {filename_ext: binary_content}
        """
        for fname, content in data.items():
            info = tarfile.TarInfo(name=f"{key}.{fname}")
            info.size = len(content)
            self._tar.addfile(info, io.BytesIO(content))
        self._current_count += 1
        self._meta["total_examples"] += 1

        if self._current_count >= self.shard_size:
            self._open_new_shard()

    def close(self) -> None:
        if self._tar:
            self._tar.close()
        # Write metadata
        meta_path = self.output_dir / "shard_meta.json"
        with open(meta_path, "w") as f:
            json.dump(dict(self._meta), f)
        logger.info(f"Shard writing complete. {self._meta['total_examples']} examples in {self._shard_idx} shards")


class WebDatasetConverter:
    """Convert Parquet/CSV to WebDataset or Arrow format for H100s."""

    def __init__(
        self,
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 256,
        num_workers: int = 8,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) if TRANSFORMERS_AVAILABLE else None
        self.max_length = max_length
        self.num_workers = num_workers

    def tokenize_example(self, text: str) -> Dict[str, np.ndarray]:
        """Tokenize text to input_ids and attention_mask."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not available")
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        return {
            "input_ids": encoded["input_ids"].astype(np.int32).squeeze(),
            "attention_mask": encoded["attention_mask"].astype(np.int32).squeeze(),
        }

    def convert_to_webdataset(
        self,
        input_path: str,
        output_dir: str,
        shard_size: int = 10000,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        compression: str = "gz",
    ) -> Dict[str, str]:
        """Convert dataframe to sharded WebDataset.

        Returns:
            Dict mapping split name to output directory.
        """
        if not WEBDATASET_AVAILABLE:
            raise RuntimeError("webdataset not installed")

        df = self._load(input_path)
        logger.info(f"Loaded {len(df)} rows. Converting to WebDataset shards...")

        # Deterministic shuffle and split
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(df)
        train_end = int(n * split_ratios[0])
        val_end = train_end + int(n * split_ratios[1])

        splits = {
            "train": df.iloc[:train_end],
            "val": df.iloc[train_end:val_end],
            "test": df.iloc[val_end:],
        }

        result = {}
        for split_name, split_df in splits.items():
            split_dir = Path(output_dir) / split_name
            writer = ShardWriter(
                str(split_dir),
                shard_size=shard_size,
                compress=True,
                compression=compression,
            )

            for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"{split_name}"):
                text = str(row.get("combined_text", row.get("text", "")))
                label = int(row.get("label", 0))

                tokens = self.tokenize_example(text)

                # Serialize to binary
                data = {
                    "input_ids": tokens["input_ids"].tobytes(),
                    "attention_mask": tokens["attention_mask"].tobytes(),
                    "label": np.array([label], dtype=np.int64).tobytes(),
                    "json": json.dumps({
                        "label": label,
                        "label_score": int(row.get("label_score", 0)),
                        "subreddit": str(row.get("subreddit", "")),
                    }).encode(),
                }
                writer.write(f"{split_name}_{idx:08d}", data)

            writer.close()
            result[split_name] = str(split_dir)

        return result

    def convert_to_arrow(
        self,
        input_path: str,
        output_path: str,
        pre_tokenize: bool = True,
    ) -> str:
        """Convert to memory-mappable Arrow IPC file.

        Best for single-node training where you want OS-level caching.
        """
        df = self._load(input_path)
        logger.info(f"Converting {len(df)} rows to Arrow IPC format")

        if pre_tokenize and self.tokenizer:
            input_ids_list = []
            attention_mask_list = []
            for text in tqdm(df["combined_text"], desc="Tokenizing"):
                tokens = self.tokenize_example(str(text))
                input_ids_list.append(tokens["input_ids"])
                attention_mask_list.append(tokens["attention_mask"])

            df = df.copy()
            df["input_ids"] = input_ids_list
            df["attention_mask"] = attention_mask_list

        table = pa.Table.from_pandas(df)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with pa.ipc.new_file(output_path, schema=table.schema) as writer:
            writer.write_table(table)

        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        logger.info(f"Arrow IPC written: {output_path} ({size_mb:.1f} MB)")
        return output_path

    def _load(self, path: str) -> pd.DataFrame:
        path = Path(path)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".csv":
            return pd.read_csv(path)
        elif path.suffix in (".arrow", ".ipc"):
            import pyarrow.ipc as ipc
            with ipc.open_file(path) as reader:
                return reader.read_pandas()
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")


class DoomWebDataset(torch.utils.data.IterableDataset):
    """PyTorch IterableDataset over WebDataset shards for H100 training."""

    def __init__(
        self,
        shard_dir: str,
        shuffle_shards: bool = True,
        shuffle_buffer: int = 1000,
        num_workers: int = 4,
    ):
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.shuffle_shards = shuffle_shards
        self.shuffle_buffer = shuffle_buffer
        self.num_workers = num_workers
        self._urls = [str(p) for p in self.shard_dir.glob("shard_*.tar*")]
        if not self._urls:
            raise FileNotFoundError(f"No shards found in {shard_dir}")

    def __iter__(self):
        dataset = wds.WebDataset(self._urls, shardshuffle=self.shuffle_shards, nodesplitter=wds.split_by_node)
        dataset = dataset.shuffle(self.shuffle_buffer)
        dataset = dataset.decode("l")
        dataset = dataset.to_tuple("input_ids", "attention_mask", "label")
        dataset = dataset.map(self._to_tensor)
        for sample in dataset:
            yield sample

    @staticmethod
    def _to_tensor(input_ids_bytes, attention_mask_bytes, label_bytes):
        import torch
        input_ids = torch.frombuffer(input_ids_bytes, dtype=torch.int32).clone()
        attention_mask = torch.frombuffer(attention_mask_bytes, dtype=torch.int32).clone()
        label = torch.frombuffer(label_bytes, dtype=torch.int64).clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
        }


def main():
    parser = argparse.ArgumentParser(description="WebDataset/Arrow Converter for H100s")
    parser.add_argument("--input", required=True, help="Input Parquet/CSV")
    parser.add_argument("--output", required=True, help="Output directory or .arrow file")
    parser.add_argument("--format", choices=["webdataset", "arrow"], default="webdataset")
    parser.add_argument("--tokenizer", default="distilbert-base-uncased")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--compression", default="gz", choices=["gz", "bz2", "xz", "zstd"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    converter = WebDatasetConverter(
        tokenizer_name=args.tokenizer,
        max_length=args.max_len,
        num_workers=args.workers,
    )

    if args.format == "webdataset":
        result = converter.convert_to_webdataset(
            args.input, args.output, shard_size=args.shard_size, compression=args.compression
        )
        for split, path in result.items():
            print(f"{split}: {path}")
    else:
        converter.convert_to_arrow(args.input, args.output, pre_tokenize=True)


if __name__ == "__main__":
    import torch
    main()
