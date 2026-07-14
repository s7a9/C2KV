"""Evaluate full-length residual-QKV PIC on the repository's MDQA datasets."""

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from mdocdataset import load_mdoc_dataset
from models import get_model_class
from models.pic_utils import build_pic_cache
from reuse_pipeline import prefill_kv_cache, tokenize_for_reuse


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, float]:
    dataset = load_mdoc_dataset(
        args.dataset,
        args.dataset_path,
        only_supporting=args.only_supporting,
        enable_cot=args.cot,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    _, model_class = get_model_class(args.model, "pic")
    model = model_class.from_pretrained(
        args.model,
        device_map="auto",
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).eval()
    if not model.config.pic_enabled:
        raise ValueError(f"{args.model} is not a residual-QKV PIC checkpoint")
    device = model.device

    default_system = dataset.system_prompt or "You are a helpful assistant."
    default_system_ids = tokenize_for_reuse(
        tokenizer, [default_system], keep_bos=True, role="system"
    ).to(device)
    default_system_cache = prefill_kv_cache(model, default_system_ids)

    scores = []
    results = []
    example_count = len(dataset) if args.max_examples is None else min(args.max_examples, len(dataset))
    for index in tqdm(range(example_count)):
        example = dataset[index]
        system_cache = default_system_cache
        if "system_prompt" in example:
            system_ids = tokenize_for_reuse(
                tokenizer, [example["system_prompt"]], keep_bos=True, role="system"
            ).to(device)
            system_cache = prefill_kv_cache(model, system_ids)
        system_length = system_cache.get_seq_length()
        system_mask = torch.ones((1, system_length), dtype=torch.bool, device=device)

        context = tokenize_for_reuse(
            tokenizer, example["documents"], keep_bos=False, role="user"
        ).to(device)
        context_ids = context.input_ids.masked_fill(~context.attention_mask.bool(), -100)
        model.model.config._attn_implementation = "flex_attention"
        pic_cache, cache_mask, context_lengths = build_pic_cache(
            model.model,
            context_ids.unsqueeze(0),
            past_key_values=system_cache,
            past_attention_mask=system_mask,
        )

        query_ids = tokenize_for_reuse(
            tokenizer,
            [example["question"]],
            keep_bos=False,
            role="user",
            add_generation_prompt=True,
        ).input_ids.to(device)
        query_length = query_ids.shape[1]
        context_length = int(context_lengths[0].item())
        start_position = system_length + context_length
        position_ids = torch.arange(
            start_position, start_position + query_length, device=device
        ).unsqueeze(0)

        # GenerationMixin expects input_ids/attention_mask to describe the cached prefix.
        prefix_ids = torch.zeros((1, pic_cache.get_seq_length()), dtype=torch.long, device=device)
        generation_ids = torch.cat([prefix_ids, query_ids], dim=1)
        attention_mask = torch.cat(
            [cache_mask, torch.ones_like(query_ids, dtype=torch.bool)], dim=1
        )
        model.model.config._attn_implementation = "flash_attention_2"
        generated = model.generate(
            input_ids=generation_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=pic_cache,
            max_new_tokens=example.get("max_new_tokens", dataset.max_new_tokens),
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            use_pic=True,
        )
        prediction = tokenizer.decode(
            generated[0, generation_ids.shape[1]:], skip_special_tokens=True
        )
        score = dataset.metric(prediction, example["answer"])
        scores.append(score)
        results.append(
            {
                "qid": example["qid"],
                "prediction": prediction,
                "ground_truth": example["answer"],
                "score": score,
                "context_tokens": context_length,
                "cached_context_tokens": context_length,
            }
        )

    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as output:
            for result in results:
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
    return {"score": sum(scores) / len(scores) if scores else 0.0, "examples": len(scores)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate residual-QKV PIC on MDQA")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_path")
    parser.add_argument("--output_file")
    parser.add_argument("--max_examples", type=int)
    parser.add_argument("--only_supporting", action="store_true")
    parser.add_argument("--cot", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
