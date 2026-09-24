#!/usr/bin/env python3
"""
Build a noise-augmented SLURP dataset (train/devel/test jsonl + audio_root)
in the exact format train.py / dataset.py expect:

    slurp_jsonl_dir/{train,devel,test}.jsonl   -- each entry's "recordings"
                                                   list has one file per SNR
    audio_root/slurp_real/*, audio_root/slurp_synth/*  -- SNR-tagged files

Each original recording "audio-id.flac" becomes 3 entries in the same
sentence's "recordings" list:
    audio-id__0dB.flac, audio-id__5dB.flac, audio-id__10dB.flac
so SlurpDataset (which flattens "recordings" into one example per file)
naturally trains on all three SNR copies of every utterance.

Usage:
    python build_slurp_noisy_manifest.py \
        --slurp-metadata /path/to/original/slurp/dataset/slurp \
        --snr-dirs 0dB=/data/0dB 5dB=/data/5dB 10dB=/data/10dB \
        --output-jsonl-dir /path/to/new/slurp/dataset/slurp \
        --audio-root /path/to/new/slurp/audio \
        --link-mode symlink
"""

import argparse
import json
import os
import shutil
import sys

SPLITS = ["train", "devel", "test"]


def parse_snr_dirs(pairs):
    snr_map = {}
    for pair in pairs:
        if "=" not in pair:
            sys.exit(f"Invalid --snr-dirs entry '{pair}'. Use LABEL=PATH, e.g. 0dB=/data/0dB")
        label, path = pair.split("=", 1)
        if not os.path.isdir(path):
            sys.exit(f"Directory not found for '{label}': {path}")
        snr_map[label] = path
    return snr_map


def build_file_index(snr_dir):
    """Map base filename -> (full_path, subdir_name), where subdir_name is
    'slurp_real' / 'slurp_synth' if that's where the file lives under
    snr_dir, else None."""
    index = {}
    for root, _, files in os.walk(snr_dir):
        rel = os.path.relpath(root, snr_dir)
        subdir = None
        for tag in ("slurp_real", "slurp_synth"):
            if tag in rel.split(os.sep):
                subdir = tag
                break
        for f in files:
            index[f] = (os.path.join(root, f), subdir)
    return index


def place_file(src_path, dest_path, link_mode):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        return
    if link_mode == "symlink":
        os.symlink(os.path.abspath(src_path), dest_path)
    elif link_mode == "hardlink":
        os.link(src_path, dest_path)
    else:  # copy
        shutil.copy2(src_path, dest_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slurp-metadata", required=True,
                     help="Directory containing the ORIGINAL train.jsonl / devel.jsonl / test.jsonl")
    ap.add_argument("--snr-dirs", required=True, nargs="+",
                     help="LABEL=PATH pairs, e.g. 0dB=/data/0dB 5dB=/data/5dB 10dB=/data/10dB")
    ap.add_argument("--output-jsonl-dir", required=True,
                     help="Where to write the new train.jsonl / devel.jsonl / test.jsonl")
    ap.add_argument("--audio-root", required=True,
                     help="Where to place SNR-tagged audio (creates slurp_real/ and slurp_synth/ inside)")
    ap.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink",
                     help="How to place audio into --audio-root. symlink is fastest and saves disk space.")
    ap.add_argument("--splits", nargs="+", default=SPLITS,
                     help="Which split files to process (default: train devel test)")
    ap.add_argument("--strict", action="store_true",
                     help="Fail if any recording is missing from an SNR folder instead of skipping it.")
    args = ap.parse_args()

    snr_dirs = parse_snr_dirs(args.snr_dirs)

    print("Indexing audio files...")
    snr_index = {label: build_file_index(path) for label, path in snr_dirs.items()}
    for label, idx in snr_index.items():
        print(f"  {label}: {len(idx)} files found in {snr_dirs[label]}")

    os.makedirs(args.output_jsonl_dir, exist_ok=True)

    n_missing = 0
    for split in args.splits:
        in_path = os.path.join(args.slurp_metadata, f"{split}.jsonl")
        if not os.path.exists(in_path):
            print(f"  (skipping {split}: {in_path} not found)")
            continue

        out_path = os.path.join(args.output_jsonl_dir, f"{split}.jsonl")
        n_entries = 0
        n_recordings = 0

        with open(in_path, "r", encoding="utf-8") as in_f, \
             open(out_path, "w", encoding="utf-8") as out_f:
            for line in in_f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                recordings = entry.get("recordings", [])
                if not recordings:
                    continue

                new_recordings = []
                for rec in recordings:
                    orig_filename = rec["file"]
                    base, ext = os.path.splitext(orig_filename)

                    for snr_label, index in snr_index.items():
                        hit = index.get(orig_filename)
                        if hit is None:
                            n_missing += 1
                            msg = (f"  MISSING [{snr_label}]: {orig_filename} "
                                   f"(slurp_id={entry.get('slurp_id')})")
                            if args.strict:
                                sys.exit(msg)
                            continue

                        src_path, subdir = hit
                        subdir = subdir or "slurp_real"  # fallback if SNR dirs aren't split
                        new_filename = f"{base}__{snr_label}{ext}"
                        dest_path = os.path.join(args.audio_root, subdir, new_filename)
                        place_file(src_path, dest_path, args.link_mode)

                        new_recordings.append({**rec, "file": new_filename})
                        n_recordings += 1

                if not new_recordings:
                    continue

                new_entry = {**entry, "recordings": new_recordings}
                out_f.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
                n_entries += 1

        print(f"[{split}] wrote {n_entries} sentences, {n_recordings} recordings -> {out_path}")

    print(f"\nAudio placed under {args.audio_root}/slurp_real and /slurp_synth "
          f"(mode: {args.link_mode})")
    if n_missing:
        print(f"Warning: {n_missing} recording/SNR combinations were missing and skipped. "
              f"Re-run with --strict to fail on missing files instead.")

    print("\nTrain with:")
    print(f"  python train.py \\\n"
          f"      --slurp_jsonl_dir {args.output_jsonl_dir} \\\n"
          f"      --audio_root {args.audio_root} \\\n"
          f"      --output_dir ./ckpt")


if __name__ == "__main__":
    main()
