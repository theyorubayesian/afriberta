import argparse
import logging
from collections import Counter
from pathlib import Path

from transformers import XLMRobertaTokenizer

MAX_LENGTH = 512
MIN_NUM_TOKENS = 5

# TODO: Update this script to take a list of languages
# and calculate overlap over all possible pairs

def tokenize_dataset(
    tokenizer: XLMRobertaTokenizer, 
    data: Path, 
    min_num_tokens: int,
    write_output: bool = True,
    output_file: Path = None
) -> Counter:
    token_counter = Counter()
    
    for line in data.read_text(encoding="utf-8").splitlines():
        if (len(line.split()) > min_num_tokens and not line.isspace()):
            tokens = tokenizer.tokenize(line)
            token_counter.update(tokens)

    tokens = sorted(
        token_counter.items(), 
        key=lambda item: (item[1], item[0]),
        reverse=True
    )
    
    if write_output:
        with open(output_file, "w") as out:
            out.writelines([f'{token}\t{count}\n' for token, count in tokens])
    return token_counter


def main():
    parser = argparse.ArgumentParser(
        description="Perform token overlap analysis for pairs of dataset")
    parser.add_argument(
        "--lang_file_1", help="File with name `train.*` where `*` is language")
    parser.add_argument(
        "--lang_file_2", help="File with name `train.*` where `*` is language")
    parser.add_argument("--tokenizer_path")
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--min_num_tokens", type=int, default=MIN_NUM_TOKENS)
    parser.add_argument("--output_dir")
    parser.add_argument("--logfile")
    args = parser.parse_args()
    
    lang_1 = args.lang_file_1.split('.')[-1]
    lang_2 = args.lang_file_2.split('.')[-1]

    if not args.output_dir:
        args.output_dir = f"token_analysis/{lang_1}_{lang_2}_overlap"

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=args.logfile, # f"{output_dir}/log.txt",
        format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.model_max_length = args.max_length
    logging.info(f"Tokenizer loaded from {args.tokenizer_path}")
    logging.info(f"Number of tokens in tokenizer: {tokenizer.vocab_size}")

    lang_1_data = Path(args.lang_file_1) 
    lang_2_data = Path(args.lang_file_2)
    logging.info(f"Datasets: {lang_1_data.name, lang_2_data.name}")

    lang_1_tokens = tokenize_dataset(
        tokenizer, 
        lang_1_data, 
        args.min_num_tokens, 
        write_output=True, 
        output_file=output_dir / f"tokens.{lang_1}"
    )
    lang_2_tokens = tokenize_dataset(
        tokenizer, 
        lang_2_data, 
        args.min_num_tokens, 
        write_output=True, 
        output_file=output_dir / f"tokens.{lang_2}"
    )
    
    intersection = sorted(
        (lang_1_tokens.keys() & lang_2_tokens.keys()), 
        key=lambda x: lang_1_tokens[x] + lang_2_tokens[x],
        reverse=True
    )

    overlap_with_lang_1 = len(intersection) / len(lang_1_tokens)
    overlap_with_lang_2 = len(intersection) / len(lang_2_tokens)

    logger.info(f"Number of tokens in {lang_1_data.name}: {len(lang_1_tokens)}")
    logger.info(f"Number of tokens in {lang_2_data.name}: {len(lang_2_tokens)}")
    logger.info(f"Number of intersecting tokens: {len(intersection)}")
    logger.info(f"Overlap with {lang_1_data.name}: {overlap_with_lang_1}")
    logger.info(f"Overlap with {lang_2_data.name}: {overlap_with_lang_2}")

    output_file = output_dir / "overlap.txt"
    with open(output_file, "w") as out:
        out.writelines([token + "\n" for token in intersection])


if __name__ == "__main__":
    main()
