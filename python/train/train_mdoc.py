import os
import logging
from transformers import HfArgumentParser, DataCollatorWithPadding

from .train_data import get_dataset
from .pic_trainer import PICMultiDocTrainer
from models import *
from pic_args import PICModelArgs, PICTrainingArgs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser([PICModelArgs, PICTrainingArgs])
    model_args, training_args = parser.parse_args_into_dataclasses()

    if not model_args.enable_pic:
        raise ValueError("train.train_mdoc now implements residual-QKV PIC; pass --enable_pic True")
    if model_args.attn_impl != "flash_attention_2":
        raise ValueError("Residual-QKV PIC training requires --attn_impl flash_attention_2")
    if training_args.train_data is None:
        raise ValueError("--train_data is required")
    if model_args.pic_gradient_checkpointing:
        import models.pic_utils as _pic_utils
        _pic_utils.PIC_GRADIENT_CHECKPOINTING = True

    model, tokenizer = get_model_and_tokenizer(model_args, evaluation_mode=not training_args.do_train)

    if training_args.only_train_pic:
        for name, param in model.named_parameters():
            param.requires_grad_('residual_' in name)

    if model.config.model_type != "qwen3":
        raise ValueError(f"Residual-QKV PIC is implemented only for Qwen3, got {model.config.model_type}")
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    if not trainable_names:
        raise RuntimeError("No trainable residual QKV parameters were found")
    if training_args.only_train_pic and any('residual_' not in name for name in trainable_names):
        raise RuntimeError("Base-model parameters unexpectedly remain trainable")
    
    logger.info(f"Total Model params: {format_numel_str(sum(p.numel() for p in model.parameters()))}")
    logger.info(f"Trainable Model params: {format_numel_str(sum(p.numel() for p in model.parameters() if p.requires_grad))}")

    with training_args.main_process_first(desc="Get dataset"):
        dataset_args = {
            'tokenizer': tokenizer, 'shuffle_seed': training_args.dataset_shuffle_seed,
            'max_doc_length': 1024, 'max_doc_num': 10, 'max_length': 1024,
            'max_system_length': 256, 'dynamic_context_cap': 4096,
        }
        hotpotqa_path = os.path.join(training_args.train_data + "_cleaned", "hotpotqa_train_cleaned")
        wikimqa_path = os.path.join(training_args.train_data + "_cleaned", "wikimqa_train_cleaned")
        longmagpie_path = os.path.join(training_args.train_data, "longmagpie_1024")
        nextcoder_path = os.path.join(training_args.train_data, "microsoft--NextCoderDataset")

        train_dataset = get_dataset('mdoc', hotpotqa_path, **dataset_args)
        wikimqa_train = get_dataset('mdoc', wikimqa_path, **dataset_args)
        tulu3_train = get_dataset('mdoc', "allenai/tulu-3-sft-mixture", **dataset_args)
        nextcoder_train = get_dataset('mdoc', nextcoder_path, **dataset_args)
        longmagpie_train = get_dataset('mdoc', longmagpie_path, **dataset_args)

        eval_dataset = get_dataset('mdoc_eval', hotpotqa_path, **dataset_args)
        wikimqa_eval = get_dataset('mdoc_eval', wikimqa_path, **dataset_args)

        for subset, limit in (
            (train_dataset, 40000),
            (wikimqa_train, 40000),
            (tulu3_train, 80000),
            (nextcoder_train, 56000),
            (longmagpie_train, 40000),
        ):
            subset.data = subset.data.select(range(min(limit, len(subset.data))))

        train_dataset.merge([wikimqa_train, tulu3_train, nextcoder_train, longmagpie_train])
        eval_dataset.merge([wikimqa_eval], method='concat')

    trainer = PICMultiDocTrainer(
        model=model,
        args=training_args,
        max_doc_length=train_dataset.max_doc_length,
        model_args=model_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer, 
            padding=True,
            return_tensors='pt',
        ),
    )

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
    else:
        eval_result = trainer.evaluate()
        with training_args.main_process_first(desc="Evaluate model"):
            logger.info(f"Evaluation result: {eval_result}")
    
if __name__ == "__main__":
    main()
