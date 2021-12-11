import argparse

import sentencepiece as spm

SPECIAL_TOKENS_ID = {
    "BOS": 0,
    "PAD": 1,
    "EOS": 2,
    "UNK": 3,
}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Learn Subword")
    parser.add_argument("--input", type=str, required=True, help="Path of input file")
    parser.add_argument(
        "--model_prefix", type=str, required=True, help="Name of model prefix",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=8004, help="Vocabulary size of subword model",
    )
    parser.add_argument(
        "--max_sent_len", type=int, default=100000, help="Max sentence length in bytes",
    )
    parser.add_argument(
        "--character_coverage", type=float, default=0.9995, help="Character coverage",
    )
    parser.add_argument(
        "--lowercase", type=str, default="False", help="Lowercase while learning subwords",
    )
    parser.add_argument(
        "--split_digits",
        type=str,
        default="True",
        help="split all digits (0-9) into separate pieces",
    )
    parser.add_argument(
        "--byte_fallback",
        type=str,
        default="True",
        help="decompose unknown pieces into UTF-8 byte pieces",
    )
    parser.add_argument(
        "--split_by_whitespace", type=str, default="True", help="Split pieces by whitespace",
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()

    normalization_rule_name = "nmt_nfkc"
    if eval(params.lowercase.title()):
        # this normalization rule takes care of case folding (https://github.com/google/sentencepiece/blob/master/doc/normalization.md)
        normalization_rule_name = "nmt_nfkc_cf"

    spm.SentencePieceTrainer.Train(
        input=params.input,
        model_prefix=params.model_prefix,
        vocab_size=params.vocab_size,
        normalization_rule_name=normalization_rule_name,
        character_coverage=params.character_coverage,
        max_sentence_length=params.max_sent_len,
        byte_fallback=eval(params.byte_fallback),
        split_by_whitespace=eval(params.split_by_whitespace),
        split_digits=eval(params.split_digits),
        bos_id=SPECIAL_TOKENS_ID["BOS"],
        eos_id=SPECIAL_TOKENS_ID["EOS"],
        pad_id=SPECIAL_TOKENS_ID["PAD"],
        unk_id=SPECIAL_TOKENS_ID["UNK"],
    )
