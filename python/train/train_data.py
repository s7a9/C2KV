import os
import datasets
import glob
from transformers import AutoTokenizer
from typing import Dict, List, Any, Mapping, Optional, Callable, Iterator
from logging import getLogger

from .configs import QA_QUERY_PROMPTS, CONTINUATION_PROMPTS
from inference.mdocdataset import load_mdoc_dataset, QA_SYSTEM_PROMPT

logger = getLogger(__name__)

MAX_CHUNKS = 10
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def add_eos(inputs: Mapping, eos_token_id: int):
    """Add eos for BatchEncoding object."""
    assert isinstance(inputs["input_ids"], list), f"Make sure the return_tensors are set to list!"
    if inputs["input_ids"][-1] != eos_token_id:
        for k, v in inputs.items():
            if k in ["input_ids", "labels"]:
                v = v + [eos_token_id]
            elif k == "attention_mask":
                v = v + [1]
            elif k == "position_ids":
                v = v + [v[-1] + 1]
            elif k == "token_type_ids":
                v = v + v[-1:]
            else:
                raise NotImplementedError(f"Inputs key {k} not implemented!")
            inputs[k] = v
    return inputs

def get_data_files(path: str, split: str) -> List[str]:
    filename = os.path.basename(path) + '_cached_data_files.txt'
    cached_data_files = os.path.join('/tmp', filename)
    if os.path.exists(cached_data_files):
        with open(cached_data_files, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    data_files = []
    with open(cached_data_files, 'w') as f:
        for file in glob.iglob(os.path.join(path, split, '**'), recursive=True):
            if '.' in file:
                data_files.append(file)
                f.write(file + '\n')
    return data_files

def tokenize(
    tokenizer: AutoTokenizer, 
    text: str, 
    role: str, 
    max_length: int | None = None,
    keep_bos: bool = False,
    add_generation_prompt: bool = False,
):
    if not keep_bos and tokenizer.bos_token is not None and max_length is not None:
        # the bos token is not counted in max_length
        max_length = max_length + 1
    input_ids = tokenizer.apply_chat_template(
        [{"role": role, "content": text}], tokenize=True, 
        max_length=max_length, truncation=max_length is not None,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    ).input_ids
    if not keep_bos and input_ids[0] == tokenizer.bos_token_id:
        input_ids = input_ids[1:]
    return input_ids


class GistDataset:
    def __init__(self, data: datasets.Dataset):
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> Dict[str, Any]:
        return self.data[index]
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.data)
    
    def select(self, slicing: slice) -> None:
        self.data = self.data.select(slicing)
    
    def merge(self, others: List["GistDataset"], method: str = 'interleave') -> "GistDataset":
        data_list: List[datasets.Dataset] = [self.data] + [other.data for other in others]
        if method == 'interleave':
            weights = [len(data) for data in data_list]
            probabilities = [weight / sum(weights) for weight in weights]
            self.data = datasets.interleave_datasets(
                data_list, probabilities=probabilities,
                stopping_strategy="all_exhausted_without_replacement",
            )
        elif method == 'concat':
            self.data = datasets.concatenate_datasets(data_list)
        else:
            raise NotImplementedError(f"Method {method} not implemented!")
        return self


class PretrainDataset(GistDataset):
    def __init__(self, 
        path: str, 
        tokenizer: AutoTokenizer = None, 
        split: str = 'train', 
        shuffle_seed: int = 42,
        min_length: int = 1024,
        max_length: int = 4096,
        num_samples: int = 2**15,
        cut_long_seq: bool = False,
        streaming: bool = True,
    ):
        self.streaming = streaming
        shuffle_seed += int(os.environ.get("LOCAL_RANK", 0))
        # dataset = datasets.load_dataset(path, split=split, streaming=True)
        # NOTE: package datasets is modified to avoid globbing on nas, which costs hours
        map_args = {
            "fn_kwargs": {
                'tokenizer': tokenizer,
                'min_length': min_length,
                'max_length': max_length,
                'cut_long_seq': cut_long_seq    
            },
            "batched": True, "batch_size": 32,
            "remove_columns": ["text", "meta"],
        }
        if streaming: # pretty hugh to load all data
            data_files = get_data_files(path, split)
            data = datasets.load_dataset(path, data_files=data_files, streaming=True)['train']
        else: # load a preprocessed subset
            data = datasets.load_from_disk(path)
            map_args['num_proc'] = 32
        self.data = data.map(self._preprocess_pretrain_data, **map_args).shuffle(seed=shuffle_seed)
        self.iterator = iter(self.data)
        self.num_samples = num_samples
        self.cached_data: Dict[int, Dict[str, List[Any]]] = {}
        self.min_length, self.max_length = min_length, max_length

    @staticmethod
    def _preprocess_pretrain_data(
        data: Dict[str, List[Any]], 
        tokenizer: AutoTokenizer,
        min_length: int,
        max_length: int,
        cut_long_seq: bool,
    ) -> Dict[str, List[Any]]:
        outputs = {'input_ids': [], "length": []}
        for text in data['text']:
            if cut_long_seq:
                encoded = tokenizer(text, max_length=max_length, truncation=True)
            else:
                encoded = tokenizer(text)
            seq_len = len(encoded["input_ids"])
            for start in range(0, seq_len - min_length, max_length):
                chunk_len = min(max_length, seq_len - start)
                if chunk_len < max_length:
                    chunk_len += 1
                    encoded = add_eos(encoded, tokenizer.eos_token_id)
                for k, v in encoded.items():
                    if k in outputs:
                        outputs[k].append(v[start:start + chunk_len])
                outputs["length"].append(chunk_len)
        return outputs
    
    def __iter__(self):
        return iter(self.data)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        if index in self.cached_data:
            return self.cached_data[index]
        sample = next(self.iterator)
        self.cached_data[index] = sample
        return sample
    
    def to_mdoc_format(self, tokenizer: AutoTokenizer, mdoc_dataset: "MultiDocDataset") -> GistDataset:
        assert not self.streaming, "Streaming is not supported when to_mdoc_format!"
        doc_length = mdoc_dataset.max_doc_length
        if self.min_length <= doc_length:
            raise ValueError(f"min_length {self.min_length} <= doc_length {doc_length}")
        def _mdoc_formatter(sample: Dict[str, Any]) -> Dict[str, Any]:
            input_ids: List[int] = sample['input_ids']
            original_length = len(input_ids)
            context_length = ((original_length - 1) // doc_length) * doc_length
            context_input_ids = input_ids[:context_length]
            context_pad_length = doc_length * mdoc_dataset.max_doc_num - context_length
            context_input_ids.extend([-100] * context_pad_length)
            input_ids = input_ids[context_length:context_length + mdoc_dataset.max_length]
            input_length = len(input_ids)
            input_pad_length = mdoc_dataset.max_length - input_length
            labels = input_ids + [-100] * input_pad_length
            input_ids.extend([tokenizer.pad_token_id] * input_pad_length)
            attention_mask = [1] * input_length + [0] * input_pad_length
            return {
                'context_input_ids': context_input_ids,
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask
            }
        return GistDataset(self.data.map(_mdoc_formatter, batched=False, num_proc=32).select(range(self.num_samples)))

    def to_mdoc_format2(self, tokenizer: AutoTokenizer, mdoc_dataset: "MultiDocDataset") -> GistDataset:
        assert not self.streaming, "Streaming is not supported when to_mdoc_format!"
        doc_length = mdoc_dataset.max_doc_length
        if self.min_length <= doc_length:
            raise ValueError(f"min_length {self.min_length} <= doc_length {doc_length}")
        # 获取换行符的token id
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        newline_id = newline_ids[0] if newline_ids else None
        def split_by_sentences(input_ids: List[int], max_length: int) -> List[List[int]]:
            """按照换行符分割，并将句子尽可能多地放入文档中"""
            if newline_id is None:
                # 如果没有换行符，直接按长度分割
                return [input_ids[i:i+max_length] for i in range(0, len(input_ids), max_length)]
            documents = []
            current_doc = []
            current_length = 0
            # 找到所有换行符位置
            sentence_boundaries = [0]
            for i, token_id in enumerate(input_ids):
                if token_id == newline_id:
                    sentence_boundaries.append(i + 1)
            sentence_boundaries.append(len(input_ids))
            # 按句子分组到文档中
            for i in range(len(sentence_boundaries) - 1):
                sentence = input_ids[sentence_boundaries[i]:sentence_boundaries[i+1]]
                sentence_length = len(sentence)
                # 如果单个句子超过最大长度，需要分割
                if sentence_length > max_length:
                    if current_doc:
                        documents.append(current_doc)
                        current_doc = []
                        current_length = 0
                    # 分割长句子
                    for j in range(0, sentence_length, max_length):
                        documents.append(sentence[j:j+max_length])
                elif current_length + sentence_length <= max_length:
                    # 可以放入当前文档
                    current_doc.extend(sentence)
                    current_length += sentence_length
                else:
                    # 当前文档已满，开始新文档
                    if current_doc:
                        documents.append(current_doc)
                    current_doc = sentence
                    current_length = sentence_length
            return documents
        
        def _mdoc_formatter(sample: Dict[str, Any]) -> Dict[str, Any]:
            input_ids: List[int] = sample['input_ids']
            # 使用伪随机数选择提示（基于样本内容确保可复现）
            seed = sum(input_ids[:min(10, len(input_ids))]) % len(CONTINUATION_PROMPTS)
            prompt_text = CONTINUATION_PROMPTS[seed]
            # 将文本分割成文档
            documents = split_by_sentences(input_ids, doc_length)
            if len(documents) <= 1:
                # 文本太短，无法分割
                return None
            # 限制文档数量
            if len(documents) > mdoc_dataset.max_doc_num + 1:
                documents = documents[:mdoc_dataset.max_doc_num + 1]
            # 准备context documents (所有文档除了最后一个用于回答)
            context_docs = documents[:-1]
            answer_doc = documents[-1]
            # 将每个context document转换为user message
            context_input_ids = []
            for doc in context_docs:
                # 解码文档内容
                doc_text = tokenizer.decode(doc, skip_special_tokens=True)
                # 使用tokenize函数转换为user message
                doc_ids = tokenize(tokenizer, doc_text, "user", max_length=doc_length)
                context_input_ids.extend(doc_ids)
                assert len(doc_ids) <= doc_length, \
                    f"Context document length {len(context_input_ids)} > doc_length {doc_length}"
                context_input_ids.extend([-100] * (doc_length - len(doc_ids)))
            empty_doc_num = mdoc_dataset.max_doc_num - len(context_docs)
            if empty_doc_num > 0:
                context_input_ids.extend([-100] * (doc_length * empty_doc_num))

            prompt_ids = tokenize(tokenizer, prompt_text, "user", add_generation_prompt=True)
            answer_ids = answer_doc + [tokenizer.eos_token_id]
            
            # 拼接 prompt 和 answer
            input_ids = prompt_ids + answer_ids
            labels = ([-100] * len(prompt_ids)) + answer_ids
            attention_mask = [1] * len(input_ids)
            # 填充
            input_pad_length = mdoc_dataset.max_length - len(input_ids)
            if input_pad_length > 0:
                labels.extend([-100] * input_pad_length)
                attention_mask.extend([0] * input_pad_length)
                input_ids.extend([tokenizer.pad_token_id] * input_pad_length)
            else:
                labels = labels[:mdoc_dataset.max_length]
                attention_mask = attention_mask[:mdoc_dataset.max_length]
                input_ids = input_ids[:mdoc_dataset.max_length]
        
            return {
                'context_input_ids': context_input_ids,
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask
            }
        
        # 过滤掉None结果
        mapped_data = self.data.map(
            _mdoc_formatter, 
            batched=False, 
            num_proc=32,
            remove_columns=self.data.column_names
        ).filter(lambda x: x is not None, num_proc=32)
        
        return GistDataset(mapped_data.select(range(min(self.num_samples, len(mapped_data)))))

class SFTDataset(GistDataset):
    def __init__(
        self,
        path: str,
        tokenizer: AutoTokenizer,
        split: str = 'train',
        num_samples: Optional[int] = None,
        max_length: int = 4096,
        min_length: int = 1024,
        shuffle_seed: int = 42,
    ):
        self.path = path
        data = datasets.load_dataset(path, split=split)
        data = data.shuffle(seed=shuffle_seed)
        if num_samples is not None:
            data = data.select(range(num_samples))
        else:
            data = data.select(range(256, len(data)))
        self.data = data.map(
            self._preprocess_sft_data,
            fn_kwargs={
                'tokenizer': tokenizer,
                'min_length': min_length,
                'max_length': max_length,
            },
            batched=True, batch_size=32, num_proc=32,
            remove_columns=data.column_names,
        )

    @staticmethod
    def _preprocess_sft_data(
        data: Dict[str, List[Any]], 
        tokenizer: AutoTokenizer,
        min_length: int,
        max_length: int,
    ) -> Dict[str, List[Any]]:
        outputs = {'input_ids': [], 'labels': []}
        for prompt, response in zip(data['instruction'], data['output']):
            prompt_input_ids = tokenizer(prompt)["input_ids"]
            response_input_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
            response_input_ids.append(tokenizer.eos_token_id)
            req_length = len(prompt_input_ids) + len(response_input_ids)
            if req_length < min_length or req_length > max_length:
                continue
            outputs['input_ids'].append(prompt_input_ids + response_input_ids)
            label = [-100] * len(prompt_input_ids) + response_input_ids
            label.extend([-100] * (max_length - req_length)) # pad here with -100
            outputs['labels'].append(label)
        return outputs


def _is_valid_tulu_sample(sample: Dict[str, Any]) -> bool:
    """Check if a tulu sample is valid for processing."""
    messages = sample.get('messages', [])
    if not messages:
        return False

    # Filter out system messages
    messages = [msg for msg in messages if msg.get('role') != 'system']

    if len(messages) < 2:
        return False

    user_messages = [msg for msg in messages if msg.get('role') == 'user']
    assistant_messages = [msg for msg in messages if msg.get('role') == 'assistant']

    # Single-turn: check if user message has double newlines
    if len(user_messages) == 1 and len(assistant_messages) == 1:
        user_content = user_messages[0].get('content', '')
        return '\n\n' in user_content

    # For multi-turn, we require at least 2 messages and total content length >= 8192 to ensure enough context
    if sum(map(lambda msg: len(msg.get('content', '')), messages)) < 8192:
        return False

    # Multi-turn: valid if at least 2 messages
    return len(messages) >= 2


def _split_by_delimiter(text: str) -> List[str]:
    """
    Split text by hierarchical delimiters, following Block-Attention approach.
    Priority: \n\n → --- → === → \n\t → \n
    """
    import re

    # For double newlines and single newlines, pair up results (merge delimiter with content)
    if "\n\n" in text:
        parts = re.split(r'(\n\n)', text)
        if len(parts) == 1:
            return parts
        # Merge delimiter with preceding content: [content, \n\n, content, \n\n, ...]
        result = [parts[i] + parts[i+1] for i in range(0, len(parts)-1, 2)]
        if len(parts) % 2 == 1:  # Odd number means last part has no delimiter
            result.append(parts[-1])
        return result

    if "---" in text:
        return re.split(r'(---)', text)

    if "===" in text:
        return re.split(r'(===)', text)

    if "\n\t" in text:
        return re.split(r'(\n\t)', text)

    if "\n" in text:
        parts = re.split(r'(\n)', text)
        if len(parts) == 1:
            return parts
        result = [parts[i] + parts[i+1] for i in range(0, len(parts)-1, 2)]
        if len(parts) % 2 == 1:
            result.append(parts[-1])
        return result

    return [text]


def _merge_smallest_chunks(chunks: List[str]) -> List[str]:
    """Merge the two smallest adjacent chunks."""
    if len(chunks) <= 1:
        return chunks

    # Find the smallest adjacent pair
    min_size = float('inf')
    min_idx = 0
    for i in range(len(chunks) - 1):
        combined_size = len(chunks[i]) + len(chunks[i+1])
        if combined_size < min_size:
            min_size = combined_size
            min_idx = i

    # Merge the pair
    merged = chunks[:min_idx] + [chunks[min_idx] + chunks[min_idx+1]] + chunks[min_idx+2:]
    return merged


def _split_context_messages(messages: List[Dict[str, str]], max_chunks: int = MAX_CHUNKS) -> List[str]:
    """
    Split context messages into documents using hierarchical delimiter-based splitting.
    Follows Block-Attention approach with chunk limit enforcement.
    """
    # If there are many messages (≥12), don't split them - each message becomes one chunk
    if len(messages) >= 12:
        return [msg['content'] for msg in messages]

    # Split each message into chunks
    all_chunks = []
    for msg in messages:
        chunks = _split_by_delimiter(msg['content'])
        all_chunks.extend(chunks)

    # Enforce max_chunks limit by merging smallest chunks
    while len(all_chunks) > max_chunks:
        all_chunks = _merge_smallest_chunks(all_chunks)

    return all_chunks


def _split_long_documents(documents: List[str], max_doc_length: int, max_doc_num: int) -> List[str]:
    """Split documents that exceed max_doc_length (estimated as len(text) / 2 tokens)
    and cap at max_doc_num."""
    max_char_length = max_doc_length * 2
    result = []
    for doc in documents:
        if len(doc) <= max_char_length:
            result.append(doc)
        else:
            for i in range(0, len(doc), max_char_length):
                chunk = doc[i:i + max_char_length].strip()
                if chunk:
                    result.append(chunk)
    while len(result) > max_doc_num:
        result = _merge_smallest_chunks(result)
    return result


def _process_single_turn(messages: List[Dict[str, str]], max_doc_length: int, max_doc_num: int) -> Dict[str, Any]:
    """Process single-turn conversation from tulu dataset."""
    user_msg = next(msg['content'] for msg in messages if msg['role'] == 'user')
    assistant_msg = next(msg['content'] for msg in messages if msg['role'] == 'assistant')

    # Split using hierarchical delimiter approach
    chunks = _split_by_delimiter(user_msg)

    # Last chunk is question, rest are context documents
    question = chunks[-1].strip()
    documents = [c.strip() for c in chunks[:-1] if c.strip()]

    while len(documents) > max_doc_num:
        documents = _merge_smallest_chunks(documents)

    return {
        'documents': documents,
        'question': question,
        'answer': [assistant_msg]
    }


def _process_multi_turn(messages: List[Dict[str, str]], max_doc_length: int, max_doc_num: int) -> Dict[str, Any]:
    """Process multi-turn conversation from tulu dataset."""
    # Last turn is Q&A
    # Find last user and assistant messages
    last_user_idx = None
    last_assistant_idx = None

    for i in range(len(messages) - 1, -1, -1):
        if messages[i]['role'] == 'user' and last_user_idx is None:
            last_user_idx = i
        if messages[i]['role'] == 'assistant' and last_assistant_idx is None:
            last_assistant_idx = i
        if last_user_idx is not None and last_assistant_idx is not None:
            break

    question = messages[last_user_idx]['content']
    answer = messages[last_assistant_idx]['content']

    # Previous turns become context (everything before the last Q&A pair)
    last_qa_start = min(last_user_idx, last_assistant_idx)
    context_messages = messages[:last_qa_start] if last_qa_start > 0 else []

    # Convert context messages to documents
    documents = _split_context_messages(context_messages) if context_messages else []

    return {
        'documents': documents,
        'question': question,
        'answer': [answer]
    }


def _extract_nextcoder_code(prompt: str):
    """Extract instruction (before first ```) and code (inside ``` fences) from a NextCoder prompt."""
    import re
    match = re.search(r'```(?:\w*)\n(.*?)```', prompt, re.DOTALL)
    if match:
        instruction = prompt[:match.start()].strip()
        code = match.group(1).strip()
    else:
        # No code fence found — treat entire prompt as instruction
        instruction = prompt.strip()
        code = ""
    return instruction, code


def extract_nextcoder_documents(sample: Dict[str, Any], max_doc_length: int, max_doc_num: int) -> Dict[str, Any]:
    """Extract documents, question, and answer from NextCoder dataset format."""
    instruction, code = _extract_nextcoder_code(sample['prompt'])

    # Split code into chunks by double newlines
    if code:
        chunks = [c.strip() for c in code.split('\n\n') if c.strip()]
    else:
        chunks = []

    while len(chunks) > max_doc_num:
        chunks = _merge_smallest_chunks(chunks)

    chunks = _split_long_documents(chunks, max_doc_length, max_doc_num)

    return {
        'documents': chunks,
        'question': instruction,
        'answer': [sample['completion']],
    }


def extract_longalign_documents(sample: Dict[str, Any], max_doc_length: int, max_doc_num: int) -> Dict[str, Any]:
    """
    Extract documents, question, and answer from longalign-10k format.
    Each sample has 'messages' with a single user turn (long context + question)
    and a single assistant turn (answer).
    """
    messages = sample['messages']
    user_msg = next(msg['content'] for msg in messages if msg['role'] == 'user')
    assistant_msg = next(msg['content'] for msg in messages if msg['role'] == 'assistant')

    chunks = _split_by_delimiter(user_msg)

    question = chunks[-1].strip()
    documents = [c.strip() for c in chunks[:-1] if c.strip()]

    while len(documents) > max_doc_num:
        documents = _merge_smallest_chunks(documents)

    documents = _split_long_documents(documents, max_doc_length, max_doc_num)

    return {
        'documents': documents,
        'question': question,
        'answer': [assistant_msg],
    }


def extract_tulu_documents(sample: Dict[str, Any], max_doc_length: int, max_doc_num: int) -> Dict[str, Any]:
    """
    Extract documents, question, and answer from tulu-3-sft-mixture format.
    """
    messages = sample['messages']

    # Capture the first system message (if any) before filtering it out
    system_text = next((msg['content'] for msg in messages if msg['role'] == 'system'), None)

    # Filter out system messages
    messages = [msg for msg in messages if msg['role'] != 'system']

    # Determine if single-turn or multi-turn
    user_messages = [msg for msg in messages if msg['role'] == 'user']
    assistant_messages = [msg for msg in messages if msg['role'] == 'assistant']

    if len(user_messages) == 1 and len(assistant_messages) == 1:
        # Single-turn conversation
        result = _process_single_turn(messages, max_doc_length, max_doc_num)
    else:
        # Multi-turn conversation
        result = _process_multi_turn(messages, max_doc_length, max_doc_num)
    result['system'] = system_text
    return result


class MultiDocDataset(GistDataset):
    def __init__(
        self,
        path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        max_doc_length: int = 512,
        max_doc_num: int = 20,
        max_system_length: int = 256,
        dynamic_context_cap: int = 4096,
        num_samples: Optional[int] = None,
        shuffle_seed: int = 42,
    ):
        if "musique" in path:
            dataset = load_mdoc_dataset("musique", path)
            extract_documents = dataset.extract_documents
            data = dataset.data
        elif "hotpotqa" in path and "cleaned" in path:
            data = datasets.load_from_disk(path)
            extract_documents = lambda sample: sample
        elif "hotpotqa" in path:
            data = datasets.load_dataset("jsonl", data_files=path, split="train")
            extract_documents = lambda sample: sample
        elif "wikimqa" in path:
            dataset = load_mdoc_dataset("wikimqa", "xanhho/2WikiMultihopQA", split='train')
            extract_documents = lambda sample: dataset.extract_documents(sample, data_path=path)
            data = datasets.load_from_disk(path) if "cleaned" in path else dataset.data
        elif "longmagpie" in path or "longalpaca" in path:
            data = datasets.load_from_disk(path)
            extract_documents = None
        elif "nextcoder" in path.lower():
            data = datasets.load_from_disk(path)
            extract_documents = lambda s: extract_nextcoder_documents(s, max_doc_length, max_doc_num)
        elif "longalign" in path:
            data = datasets.load_dataset("json", data_files=path, split="train")
            max_context_chars = max_doc_length * max_doc_num * 4
            data = data.filter(
                lambda s: len(s['messages'][0]['content']) <= max_context_chars,
                num_proc=32,
            )
            print(len(data))
            logger.info(f"Filtered longalign dataset size: {len(data)}")
            extract_documents = lambda s: extract_longalign_documents(s, max_doc_length, max_doc_num)
        elif "tulu" in path:
            data = datasets.load_dataset(path, split="train")
            # Pre-filter invalid samples
            logger.info("Filtering tulu dataset samples...")
            data = data.filter(_is_valid_tulu_sample, num_proc=32)
            logger.info(f"Filtered dataset size: {len(data)}")
            extract_documents = lambda s: extract_tulu_documents(s, max_doc_length, max_doc_num)
        else:
            raise NotImplementedError(f"Unsupported dataset {path}")
        if num_samples is None:
            data = data.select(range(512, len(data)))
        else:
            data = data.select(range(num_samples))
        self.data = data.shuffle(seed=shuffle_seed).map(
            self._preprocess_mdoc_sample,
            fn_kwargs={
                'tokenizer': tokenizer,
                'max_length': max_length,
                'max_doc_length': max_doc_length,
                'max_doc_num': max_doc_num,
                'max_system_length': max_system_length,
                'extract_docs': extract_documents
            },
            batched=False,
            num_proc=64,
            remove_columns=data.column_names
        ).filter(lambda sample: any(token_id != -100 for token_id in sample['context_input_ids']), num_proc=64)
        self.max_doc_length = max_doc_length
        self.max_system_length = max_system_length
        self.max_doc_num = max_doc_num
        self.max_length = max_length
        self.dynamic_context_cap = dynamic_context_cap

    @staticmethod
    def _preprocess_mdoc_sample(
        sample: Dict[str, Any],
        tokenizer: AutoTokenizer,
        max_length: int,
        max_doc_length: int,
        max_doc_num: int,
        max_system_length: int,
        extract_docs: Callable | None,
    ) -> Dict[str, Any]:
        system_text = sample.get('system') or DEFAULT_SYSTEM_PROMPT
        if extract_docs is not None:
            sample = extract_docs(sample)
            sample['documents'] = [d for d in sample['documents'] if d.strip()]
            query_prompt_seed = sum(map(len, sample['documents'])) % len(QA_QUERY_PROMPTS)
            sample['question'] = QA_QUERY_PROMPTS[query_prompt_seed] + '\n\n' + sample['question']
            system_text = sample.get('system') or system_text
        concat_doc_ids = []
        for doc in sample['documents'][:max_doc_num]:
            doc_ids = tokenize(tokenizer, doc, "user", max_doc_length)
            pad_length = max_doc_length - len(doc_ids)
            assert pad_length >= 0, f"pad_length {pad_length} < 0"
            concat_doc_ids.extend(doc_ids)
            concat_doc_ids.extend([-100] * pad_length)
        concat_doc_ids.extend([-100] * (max_doc_length * (max_doc_num - len(sample['documents']))))
        system_ids = tokenize(tokenizer, system_text, "system", max_system_length, keep_bos=True)
        system_ids = system_ids[:max_system_length]
        system_input_ids = system_ids + [-100] * (max_system_length - len(system_ids))
        question_ids = tokenize(tokenizer, sample['question'], "user", add_generation_prompt=True)
        answer_ids = tokenizer.encode(sample['answer'][0], add_special_tokens=False)
        answer_ids.append(tokenizer.eos_token_id)
        input_ids = question_ids + answer_ids
        labels = [-100] * len(question_ids) + answer_ids
        pad_length = max_length - len(input_ids)
        if pad_length > 0:
            attention_mask = [1] * len(input_ids) + [0] * pad_length
            input_ids.extend([tokenizer.pad_token_id] * pad_length)
            labels.extend([-100] * pad_length)
        else:
            attention_mask = [1] * max_length
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
        return {
            'system_input_ids': system_input_ids,
            'context_input_ids': concat_doc_ids,
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'dynamic': 0,
        }


def get_dataset(dataset_type: str, path: str, tokenizer: AutoTokenizer, **kwargs) -> GistDataset:
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if dataset_type == "pretrain":
        return PretrainDataset(path, tokenizer, **kwargs)
    elif dataset_type == "pretrain_eval":
        kwargs['shuffle_seed'] = 42 # fix seed for evaluation
        kwargs['num_samples'] = kwargs.pop('num_samples', 512)
        return PretrainDataset(path, tokenizer, **kwargs)
    elif dataset_type == "sft":
        return SFTDataset(path, tokenizer, **kwargs)
    elif dataset_type == "sft_eval":
        kwargs['num_samples'] = kwargs.pop('num_samples', 256)
        return SFTDataset(path, tokenizer, **kwargs)
    elif dataset_type == "mdoc":
        return MultiDocDataset(path, tokenizer, **kwargs)
    elif dataset_type == "mdoc_eval":
        kwargs['num_samples'] = kwargs.pop('num_samples', 512)
        return MultiDocDataset(path, tokenizer, **kwargs)
    raise NotImplementedError(f"Unsupported dataset type {dataset_type}")
