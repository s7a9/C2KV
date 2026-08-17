"""
Preprocess LongMagpie dataset into multi-document QA format.

Converts the raw LongMagpie dataset (messages format) into a format suitable
for multi-document QA training with documents/question/answer columns.

Processing steps:
1. Split user message by document separator into individual documents
2. Extract the question from the end of the last document
3. Truncate long documents to ~4096 chars (~1024 tokens) using line boundaries
4. Ensure each document ends with a newline

Truncation strategy for documents exceeding 4096 chars:
- Keep the first ~4096 chars (up to the last newline before char 4096)
- Drop one paragraph (the text between the first and next newline)
- Keep the remaining text as a second chunk
- If the remaining text is empty, keep only the first chunk

Usage:
    python preprocess_longmagpie.py \\
        --input /path/to/longmagpie \\
        --output /path/to/longmagpie_1024 \\
        --max_doc_chars 4096
"""

import argparse
import logging
import os
import re
from typing import List, Tuple

import datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

DOC_SEPARATOR = "\n===Document Separator===\n\n"
MAX_DOC_CHARS = 4096

_SENTENCE_BOUNDARY = re.compile(r'[.!?][\s\u201d"]*(?=[A-Z\u201c"\n])')


def truncate_part(part: str, max_chars: int = MAX_DOC_CHARS) -> List[str]:
    """Truncate a document part to approximately max_chars using line boundaries.

    Repeatedly splits by finding the last newline before max_chars,
    dropping one paragraph (up to the next newline), and continuing
    with the remaining text until it fits within max_chars.

    Args:
        part: The text to truncate.
        max_chars: Maximum character count per chunk (approximates 1024 tokens).

    Returns:
        List of text chunks.
    """
    if len(part) <= max_chars:
        return [part]

    chunks: List[str] = []
    text = part

    while len(text) > max_chars:
        last_nl = text[:max_chars].rfind("\n")
        if last_nl < 0:
            chunks.append(text[:max_chars])
            text = ""
            break

        chunks.append(text[: last_nl + 1])
        remaining = text[last_nl + 1 :]

        next_nl = remaining.find("\n")
        if next_nl < 0:
            text = ""
            break

        text = remaining[next_nl + 1 :]

    if text.strip():
        chunks.append(text)

    return chunks


def extract_question(text: str) -> Tuple[str, int]:
    """Extract the question from the end of a text block.

    The question is appended directly after the last document text.
    We find the last sentence boundary (period/exclamation/question mark
    followed by whitespace and a capital letter) and treat everything
    after it as the question.

    Args:
        text: The text containing document content followed by a question.

    Returns:
        Tuple of (question string, character position where question starts).
    """
    boundaries = list(_SENTENCE_BOUNDARY.finditer(text))
    if not boundaries:
        return text.strip(), 0

    last = boundaries[-1]
    q_start = last.end()
    question = text[q_start:].strip()
    return question, q_start


def preprocess_sample(sample: dict, max_chars: int = MAX_DOC_CHARS) -> dict:
    """Preprocess a single LongMagpie sample.

    Args:
        sample: Raw sample with 'messages' field containing user and assistant
                messages. The user message has documents separated by
                DOC_SEPARATOR, with the question appended at the end.
        max_chars: Maximum character count per document chunk.

    Returns:
        Dict with 'documents', 'question', 'answer' fields.
    """
    messages = sample["messages"]
    user_content = messages[0]["content"]
    assistant_content = messages[1]["content"]

    parts = user_content.split(DOC_SEPARATOR)

    # Extract question from the last part
    last_part = parts[-1]
    question, q_start = extract_question(last_part)
    last_doc_text = last_part[:q_start]

    # Process all parts into document chunks
    documents: List[str] = []
    for i, part in enumerate(parts):
        if i == len(parts) - 1:
            part = last_doc_text
            if not part.strip():
                continue

        chunks = truncate_part(part, max_chars)
        documents.extend(chunks)

    # Ensure each document ends with a newline.
    # If a document ends with other punctuation (e.g., '.'), replace it with '\n'
    # to maintain consistent formatting without changing document length.
    documents = [
        doc if doc.endswith("\n") else doc[:-1] + "\n"
        for doc in documents
    ]

    return {
        "documents": documents,
        "question": question,
        "answer": [assistant_content],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess LongMagpie dataset into multi-document QA format."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/mnt/nas1/duchuheng/datasets/longmagpie",
        help="Path to the raw LongMagpie dataset directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/mnt/nas1/duchuheng/datasets/longmagpie_1024",
        help="Output path for the processed dataset.",
    )
    parser.add_argument(
        "--max_doc_chars",
        type=int,
        default=MAX_DOC_CHARS,
        help=f"Maximum characters per document chunk (default: {MAX_DOC_CHARS}).",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=64,
        help="Number of processes for parallel map.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=20,
        help="Number of shards for the output dataset.",
    )
    args = parser.parse_args()

    # Load the raw dataset from parquet files
    parquet_dir = os.path.join(args.input, "data")
    if os.path.isdir(parquet_dir):
        parquet_pattern = os.path.join(parquet_dir, "train-*.parquet")
        logger.info(f"Loading dataset from parquet files: {parquet_pattern}")
        ds = datasets.load_dataset(
            "parquet",
            data_files={"train": parquet_pattern},
            split="train",
        )
    else:
        logger.info(f"Loading dataset from: {args.input}")
        ds = datasets.load_from_disk(args.input)

    logger.info(f"Loaded {len(ds)} samples")

    logger.info(
        f"Preprocessing with max_doc_chars={args.max_doc_chars}, "
        f"num_proc={args.num_proc}"
    )

    processed = ds.map(
        lambda sample: preprocess_sample(sample, args.max_doc_chars),
        num_proc=args.num_proc,
        remove_columns=ds.column_names,
    )

    # Filter out samples with no documents or empty questions
    before_filter = len(processed)
    processed = processed.filter(
        lambda s: len(s["documents"]) > 0 and len(s["question"].strip()) > 0
    )
    logger.info(
        f"Filtered {before_filter - len(processed)} invalid samples "
        f"({before_filter} -> {len(processed)})"
    )

    logger.info(f"Saving processed dataset to: {args.output}")
    processed.save_to_disk(args.output, num_shards=args.num_shards)
    logger.info(f"Done! {len(processed)} samples saved.")


if __name__ == "__main__":
    main()
