import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path, PurePath
from types import SimpleNamespace
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import tqdm
from loguru import logger
from miditoolkit import MidiFile

from data.tokenizer import get_pretrained_tokenizer
from model import MIDIGeneratorModel
from utils import seed_others

import os
import sys
import random

def load_config(config_path):
    with open(config_path, mode="r") as f:
        data = f.read()

    json_dict = json.loads(data)
    return SimpleNamespace(**json_dict)


@eqx.filter_jit
@eqx.debug.assert_max_traces(max_traces=1)
def generate_step(model, inputs, length, key, temperature):
    logits = model(**inputs)
    logits = jnp.take(logits, length - 1, axis=0)

    if temperature == 0.0:
        # argmax sampling
        return jnp.argmax(logits, axis=-1)

    logits = logits / temperature
    return jax.random.categorical(key, logits, axis=-1)


def generate_loop(
    model,
    initial_input,
    temperature,
    key,
    max_to_generate: Optional[int] = None,
    model_max_positions: int = 1024,
    output_generated_only: bool = False,
) -> np.array:
    sample_idx = initial_input.shape[0]

    if output_generated_only:
        output = []
    else:
        output = initial_input.tolist()

    if max_to_generate is None:
        DEFAULT_MAX = 1000
        max_to_generate = DEFAULT_MAX

    input_length = sample_idx + max_to_generate
    if input_length > model_max_positions - 1:
        input_length = model_max_positions - 1

    position_ids = np.arange(input_length)
    mask = np.concatenate(
        [
            np.ones((sample_idx,), dtype=bool),
            np.zeros((input_length - sample_idx,), dtype=bool),
        ],
        axis=-1,
        dtype=bool,
    )
    input_ids = np.pad(initial_input, ((0, input_length - sample_idx),))

    # TODO: maybe replace with jax.lax.scan loop for faster generation
    # https://stackoverflow.com/questions/77364000/convert-for-loop-to-jax-lax-scan
    for i in tqdm.trange(max_to_generate, desc="Generating"):
        key, subkey = jax.random.split(key)
        inputs = dict(input_ids=input_ids, position_ids=position_ids, mask=mask)
        token = generate_step(model, inputs, np.array(sample_idx), subkey, temperature).item()
        output.append(token)

        if sample_idx < input_length:
            input_ids[sample_idx] = token
            mask[sample_idx] = True
        else:
            input_ids = np.concatenate([input_ids[1:], np.array([token])], axis=-1)

        sample_idx = min(input_length - 1, sample_idx + 1)

    return np.array(output)


def tokenize_prompt(midi, tokenizer):
    return tokenizer(midi)

def file_prompt(path):
    midi = MidiFile(path)
    return midi


def main(args):
    logger.info(os.getcwd())

    logger.info("Beginning generation script.")

    seed = 0

    if args.use_seed:
        seed = args.seed
    else:
        seed = random.randint(0, 0xFFFFFFFF)

    key = jax.random.PRNGKey(seed)
    logger.info(f"Using PRNG key {seed}")
    seed_others(seed)

    logger.info("Loading config.")
    config = load_config(Path(args.checkpoint_directory) / "config.json")

    logger.info(f"Loading tokenizer from '{args.tokenizer}'")
    tokenizer = get_pretrained_tokenizer(args.tokenizer)

    logger.info("Initialising model.")
    model = MIDIGeneratorModel(
        dim=config.dim,
        num_heads=config.heads,
        num_layers=config.num_layers,
        vocab_size=config.vocab_size,
        max_positions=config.max_sequence_length,
        head_dim=config.head_dim,
        dropout=config.dropout,
        key=key,
        dtype=jnp.bfloat16 if config.use_bf16 else jnp.float32,
    )

    if args.checkpoint_directory is None or args.checkpoint is None:
        logger.warning("Did not specify checkpoint! Using randomly initialised weights.")
        logger.warning(
            "If you do not intend to use random weights, please specifiy --checkpoint when excecuting script."
        )
    else:
        # this does not restore things like activation functions and such, so
        # we need to use a tree map later to recover these details.
        logger.info(f"Loading model from '{Path(PurePath(args.checkpoint_directory).name) / args.checkpoint}'")
        checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
        loaded_model = checkpointer.restore(
            Path(args.checkpoint_directory).resolve() / args.checkpoint,
            item=eqx.filter([model], eqx.is_inexact_array),
        )[0]

        # hack to deal with optax not serialising some equinox hyperparameters
        # TODO: change to use eqx serialisation to avoid this.
        model = jax.tree_map(lambda x, y: x if (y is None) else y, model, loaded_model)
        del loaded_model

        logger.info("Model loaded!")

    num_parameters = jax.tree_util.tree_reduce(lambda s, p: s + (p.size if eqx.is_inexact_array(p) else 0), model, 0)
    logger.info(f"Model has {num_parameters:,} parameters.")

    if args.prompt_mode == "unconditional":
        start_tokens = np.array([1], dtype=int)  # BOS token only
    elif args.prompt_mode == "file":
        logger.info(f"Loading prompt file '{args.prompt_midi}'")
        midi = file_prompt(args.prompt_midi)
        logger.info(midi)
        logger.info("Tokenising prompt.")
        start_tokens = np.array(tokenize_prompt(midi, tokenizer))[0]

    logger.info(f"Tokenised prompt is of length {start_tokens.shape[0]}")

    if args.prompt_midi_slice is not None:
        logger.info(f"Slicing starting prompt to {args.prompt_midi_slice} tokens")
        start_tokens = start_tokens[: args.prompt_midi_slice]

    if start_tokens.shape[0] >= config.max_sequence_length:
        logger.warning("Tokenised prompt provided is longer than maximum length supported by model.")
        logger.warning("Terminating")
        return

    logger.info("Beginning generation loop")
    generated_tokens = generate_loop(
        model,
        start_tokens,
        args.temperature,
        key,
        max_to_generate=args.max_to_generate,
        model_max_positions=config.max_sequence_length - 1,
        output_generated_only=args.output_generated_only,
    )

    logger.info(f"Generated MIDI has {len(generated_tokens)} tokens.")
    logger.info("Decoding generated MIDI")
    generated_midi = tokenizer(np.expand_dims(generated_tokens, axis=0))

    if args.output_file is None:
        output_dir = Path("generated_midis")
        output_dir.mkdir(exist_ok=True)
        args.output_file = output_dir / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".mid")

    logger.info(f"Saving generated MIDI to '{args.output_file}'")
    generated_midi.dump(args.output_file)
    logger.info("Done")


def validate_args(args):
    if args.checkpoint_directory is None:
        raise ValueError("Must specify --checkpoint_directory!")
    if args.checkpoint is None:
        raise ValueError("Must specify --checkpoint!")
    if args.prompt_mode not in ["unconditional", "file"]:
        raise ValueError(f"Invalid prompt mode 'args.prompt_mode'!")
    if args.prompt_mode == "file" and args.prompt_midi is None:
        raise ValueError("Must specify --prompt_midi if `--prompt_mode file` specified!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--use_seed",
        action='store_false',
        help="Use a fixed seed for PRNG key initialisation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0xFF,
        help="Random seed used for PRNG key initialisation.",
    )
    """parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to JSON config file generated by train.py",
    )"""
    parser.add_argument(
        "--checkpoint_directory",
        type=str,
        default=None,
        help="Path to directory containing trained model parameters. If not specified, generate from random weights.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Which checkpoint to use.",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="file",
        help="Specifies the prompting mode. Currently just 'file' and 'unconditional' are supported.",
    )
    parser.add_argument(
        "--prompt_midi",
        type=str,
        default=None,
        help="Path to the MIDI file to use as a prompt.",
    )
    parser.add_argument(
        "--prompt_midi_slice",
        type=int,
        default=None,
        help="Specifies the number of tokens to take from the start of the prompt file for use as a prompt.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tokenizer.json",
        help="Path to BPE tokenizer to use when tokenizing the prompt and detokenizing the sample.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. This scales the logits and can be used to influence 'how creative' the model is when generating.",
    )
    parser.add_argument(
        "--max_to_generate",
        type=int,
        default=None,
        help="Max number of tokens to generate.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="File to save result to. Defaults to creating a new file in generated_midis.",
    )
    parser.add_argument(
        "--output_generated_only",
        action="store_true",
        help="Only save the generated content to file, do not prepend the prompt.",
    )
    args = parser.parse_args()
    validate_args(args)
    main(args)
