#!/usr/bin/env python

import os
import argparse
import shutil
from pathlib import Path

from .tokenizer import get_pretrained_tokenizer, get_tokenizer, get_tokenizer_config


def main(args):
    tokenizer_config = get_tokenizer_config(
        num_velocities=args.num_velocities,
        use_chords=args.use_chords,
        use_tempos=args.use_tempos,
        use_sustain_pedal=args.use_sustain_pedal,
    )
    tokenizer = get_tokenizer(tokenizer_config)

    midi_paths = list(Path(args.midis_dir).glob("**/*.mid"))
    if len(midi_paths) == 0:
        midi_paths = list(Path(args.midis_dir).glob("**/*.midi"))

    assert len(midi_paths)

    data_augmentation_offsets = [2, 1, 1]
    no_bpe_out_dir = args.out_dir + "_no_bpe"

    tokenizer.tokenize_midi_dataset(midi_paths, no_bpe_out_dir, save_programs=False)
    # tokenizer.tokenize_midi_dataset(midi_paths, "tokenized_dataset_no_bpe", data_augment_offsets=data_augmentation_offsets, save_programs=False)

    tokenizer.learn_bpe(
        vocab_size=args.vocab_size,
        tokens_paths=list(Path(no_bpe_out_dir).glob("**/*.json")),
        start_from_empty_voc=False,
    )

    tokenizer.save_params(args.out_tokenizer)
    tokenizer.apply_bpe_to_dataset(no_bpe_out_dir, args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--midis_dir",
        type=str,
        default=os.path.join(os.path.expanduser("~"), "Projects", "MusicGenerator", "datasets", "GiantMIDI-PIano_micro", "midis"),
        help="Path to directory containing MIDI files",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="dataset_tokenized",
        help="Output directory for BPE tokenized MIDI files",
    )
    parser.add_argument(
        "--out_tokenizer",
        type=str,
        default="tokenizer.json",
        help="Output path of trained BPE tokenizer",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=10_000,
        help="Vocabulary size",
    )
    parser.add_argument(
        "--num_velocities",
        type=int,
        default=16,
        help="Number of discrete velocities (volume)",
    )
    parser.add_argument(
        "--use_chords",
        action="store_true",
        help="If present, enable chord tokens in the tokenizer",
    )
    parser.add_argument(
        "--use_tempos",
        action="store_true",
        help="If present, enable tempo tokens in the tokenizer",
    )
    parser.add_argument(
        "--use_sustain_pedal",
        action="store_true",
        help="If present, enable sustain pedal tokens in the tokenizer",
    )
    args = parser.parse_args()
    main(args)
