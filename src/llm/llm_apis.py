#!/usr/bin/env python
from __future__ import annotations

from typing import Any, Dict, Callable

import logging
import numpy as np
import pyarrow as pa
import re
import torch
from llm.llm_mol import calculate_molecular_properties


# Processor to handle different question sets
class QuestionSetProcessor:
    """
    Base class for handling answer preparation and example display for a question set.

    :param name: Name of the question set.
    :type name: str
    """
    def __init__(self, name: str="") -> None:
        self.name = name

    def prepare_answers(self, table: pa.Table) -> tuple[dict[str, list[Any]], list[bool]]:
        """
        Prepare answers for the question set from a dataset.

        :param table: Dataset containing SMILES and possibly IUPAC names.
        :type table: pa.Table
        :return: Dictionary mapping property/question names to lists of answers.
        :rtype: Dict[str, Sequence[Any]]
        """
        raise NotImplementedError

class IUPACNamingProcessor(QuestionSetProcessor):
    """
    Processor for the IUPAC naming question set.
    """
    def __init__(self, name: str="iupac_naming") -> None:
        super().__init__(name)

    def prepare_answers(self, table: pa.Table) -> tuple[dict[str, list[Any]], list[bool]]:
        """
        Prepare answers for IUPAC naming (just returns the IUPAC names) and a validity mask.

        :param table: Dataset containing IUPAC names.
        :type table: pa.Table
        :return: Tuple of (answer dict, mask) where mask is True for valid names.
        :rtype: Tuple[Dict[str, Sequence[Any]], List[bool]]
        """
        iupac_list = table.column("iupac").to_pylist()
        mask = [x is not None and str(x).strip() != "" for x in iupac_list]
        return {"iupac_name": iupac_list}, mask


class MolecularPropertiesProcessor(QuestionSetProcessor):
    """
    Processor for the molecular properties question set.
    """
    def __init__(self, name: str="molecular_properties") -> None:
        super().__init__(name)

    def prepare_answers(self, table: pa.Table) -> tuple[dict[str, list[Any]], list[bool]]:
        """
        Prepare answers for molecular properties.
        Returns a mask indicating which rows are valid (all properties present and not None/empty).
        """
        smiles = table.column("smiles").to_pylist()
        answers = calculate_molecular_properties(smiles)
        answers["iupac_name"] = table.column("iupac").to_pylist()
        # Ensure all answers are lists of strings
        for k in answers:
            answers[k] = [str(x) for x in answers[k]]
        n = len(smiles)
        mask = [all(str(answers[k][i]).strip() != "" and answers[k][i] is not None for k in answers) for i in range(n)]
        return answers, mask


class AllPropertiesProcessor(QuestionSetProcessor):
    """
    Processor for the comprehensive 'all_properties' question set.
    """
    def __init__(self, name: str="all_properties") -> None:
        super().__init__(name)

    def prepare_answers(self, table: pa.Table) -> tuple[dict[str, list[Any]], list[bool]]:
        """
        Prepare answers for the comprehensive 'all_properties' question set.
        Returns a mask indicating which rows are valid (all properties present and not None/empty).
        """
        smiles = table.column("smiles").to_pylist()
        answers = calculate_molecular_properties(smiles)
        answers["iupac_name"] = table.column("iupac").to_pylist()
        # Ensure all answers are lists of strings
        for k in answers:
            answers[k] = [str(x) for x in answers[k]]
        n = len(smiles)
        mask = [all(str(answers[k][i]).strip() != "" and answers[k][i] is not None for k in answers) for i in range(n)]
        return answers, mask


def do_evaluate(accelerator: Any, model: Any, dataloader: Any, tokenizer: Any, compute_metrics: Any, max_new_tokens: int, num_examples=None, **generation_kwargs) -> tuple[dict, list[str], list[str]]:
    """
    Run generation-based evaluation and log exact-match metric.

    :param accelerator: Accelerator instance.
    :type accelerator: Accelerator
    :param model: Model instance.
    :type model: Any
    :param dataloader: DataLoader instance.
    :type dataloader: Any
    :param tokenizer: Tokenizer instance.
    :type tokenizer: Any
    :param compute_metrics: Metrics computation function.
    :type compute_metrics: Any
    :param max_new_tokens: Maximum number of new tokens.
    :type max_new_tokens: int
    :param num_examples: Number of examples to evaluate. If None, evaluate the entire dataset.
    :type num_examples: int or None
    :param generation_kwargs: Additional keyword arguments for generation (e.g., repetition_penalty, temperature, do_sample, top_p).
    :return: Tuple of (metrics dictionary, predictions list, gold labels list).
    :rtype: tuple[dict, list[str], list[str]]
    """
    logger = logging.getLogger(__name__)

    # Filter generation kwargs based on do_sample setting
    filtered_kwargs = generation_kwargs.copy()
    do_sample = filtered_kwargs.get('do_sample', False)
            
    # If sampling is disabled, remove sampling-specific parameters to avoid warnings
    if not do_sample:
        sampling_params = ['temperature', 'top_p', 'top_k']
        for param in sampling_params:
            if param in filtered_kwargs:
                # Only remove if they're at their neutral/default values
                if param == 'temperature' and filtered_kwargs[param] == 1.0:
                    filtered_kwargs.pop(param)
                elif param == 'top_p' and filtered_kwargs[param] == 1.0:
                    filtered_kwargs.pop(param)
                elif param == 'top_k' and filtered_kwargs[param] == 0:
                    filtered_kwargs.pop(param)

    model.eval()
    num_processed = 0
    
    # Accumulate all predictions and labels for example output
    all_accumulated_preds = []
    all_accumulated_labels = []
    
    for batch in dataloader:
        batch_size = batch["input_ids"].size(0)
        
        # Handle num_examples logic - if None, process entire dataset
        if num_examples is not None and num_processed + batch_size > num_examples:
            trim = num_examples - num_processed
            for k in batch:
                batch[k] = batch[k][:trim]
            batch_size = trim
            
        with torch.no_grad():
            generated = accelerator.unwrap_model(model).generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                **filtered_kwargs
            )
        generated_padded = accelerator.pad_across_processes(
                generated, dim=1, pad_index=tokenizer.pad_token_id)
        labels_padded = accelerator.pad_across_processes(
                batch["labels"], dim=1, pad_index=-100)
        gen_all    = accelerator.gather(generated_padded)
        labels_all = accelerator.gather(labels_padded)
        
        # Accumulate for example output if main process
        if accelerator.is_main_process:
            all_accumulated_preds.append(gen_all.cpu().numpy())
            all_accumulated_labels.append(labels_all.cpu().numpy())
        
        compute_metrics((gen_all.cpu().numpy(), labels_all.cpu().numpy()), compute_result=False)
        num_processed += batch_size
        
        # Break if we've processed enough examples (only if num_examples is not None)
        if num_examples is not None and num_processed >= num_examples:
            break
            
    metrics = compute_metrics((torch.empty(0), torch.empty(0)), compute_result=True)
    
    # Initialize return values
    preds = []
    gold = []
    
    if accelerator.is_main_process and all_accumulated_preds:
        try:
            def left_pad_arrays(arrays: list[np.ndarray], pad_value: int) -> np.ndarray:
                """
                Left-pad arrays to the same length and concatenate them.
                
                :param arrays: List of numpy arrays to pad and concatenate.
                :type arrays: list[np.ndarray]
                :param pad_value: Value to use for padding.
                :type pad_value: int
                :return: Concatenated array with all sequences padded to the same length.
                :rtype: np.ndarray
                """
                if not arrays:
                    return np.array([])
                
                max_len = max(arr.shape[1] for arr in arrays)
                padded_arrays = []
                
                for arr in arrays:
                    if arr.shape[1] < max_len:
                        pad_width = max_len - arr.shape[1]
                        padded_arr = np.pad(arr, ((0, 0), (pad_width, 0)), 
                                          mode='constant', constant_values=pad_value)
                        padded_arrays.append(padded_arr)
                    else:
                        padded_arrays.append(arr)
                
                return np.concatenate(padded_arrays, axis=0)
            
            # Left-pad and concatenate predictions and labels
            concat_preds = left_pad_arrays(all_accumulated_preds, tokenizer.pad_token_id)
            concat_labels = left_pad_arrays(all_accumulated_labels, -100)
            
            # Filter out-of-range tokens and decode
            def filter_out_of_range(tokens, vocab_size, pad_token_id):
                return [t if 0 <= t < vocab_size else pad_token_id for t in tokens]
            
            filtered_preds = [filter_out_of_range(seq, tokenizer.vocab_size, tokenizer.pad_token_id) for seq in concat_preds]
            concat_labels = np.where(concat_labels != -100, concat_labels, tokenizer.pad_token_id)
            filtered_labels = [filter_out_of_range(seq, tokenizer.vocab_size, tokenizer.pad_token_id) for seq in concat_labels]
            
            # Decode predictions and labels
            preds = tokenizer.batch_decode(filtered_preds, skip_special_tokens=True)
            gold = tokenizer.batch_decode(filtered_labels, skip_special_tokens=True)
            
            # Log examples (limit to first 10 to avoid overwhelming logs)
            max_examples_to_log = min(10, len(preds))
            for i in range(max_examples_to_log):
                logger.info("\nEXAMPLE %d \n PRED: %s \n GOLD: %s\n", i, preds[i], gold[i])
                
        except Exception as e:
            logger.warning("Failed to generate example predictions from accumulated data: %s", e)
    
    model.train()
    return metrics, preds, gold


def process_single_qa(
    tok: Any,
    example: Dict[str, Any],
    max_len: int,
    max_label_len: int | None = None,
    is_train: bool = True,
    system_prompt_override: str | None = None,
    create_position_weights: bool = False,
    default_weight: float = 1.0,
    answer_weight: float = 10.0,
) -> Dict[str, Any]:
    """
    Process a single question-answer pair from the expanded dataset.

    :param tok: Tokenizer instance
    :type tok: Any
    :param example: Dictionary containing a single Q&A pair with all necessary fields
    :type example: Dict[str, Any]
    :param max_len: Maximum length for the input
    :type max_len: int
    :param max_label_len: Maximum length for the label (only used for eval lable not train label)
    :type max_label_len: int or None
    :param is_train: Whether this is for training or evaluation
    :type is_train: bool
    :param system_prompt_override: Optional system prompt string. If provided, this is used instead of ``example['system_prompt']`` (which may be absent when loading datasets created without the system_prompt column).
    :type system_prompt_override: str or None
    :param create_position_weights: Whether to create position weights based on answer tags
    :type create_position_weights: bool
    :param default_weight: Weight for positions outside answer tags
    :type default_weight: float
    :param answer_weight: Weight for positions inside answer tags
    :type answer_weight: float
    :return: Dictionary with tokenized input_ids, attention_mask, labels, and optionally position_weights
    :rtype: Dict[str, Any]
    """
    # Get the EOS token from the tokenizer
    eos_token = tok.eos_token if tok.eos_token is not None else ""
    
    # Build the prompt
    if is_train:
        # For training, we include both question and answer
        if hasattr(tok, 'apply_chat_template'):
            # Use chat template if available
            prompt = [
                {"role": "system", "content": system_prompt_override if system_prompt_override is not None else example["system_prompt"]},
                {"role": "user", "content": example["question_template"].format(**example['metadata'])},
                {"role": "assistant", "content": example["assistant_template"].format(**example['metadata']) + eos_token}
            ]
            # generation prompt is an extra assistant prompt added to the end of the prompt to get the model to generate an answer
            # it's not needed for training
            prompt_str = tok.apply_chat_template(prompt, add_generation_prompt=False, tokenize=False, enable_thinking=False)
        else:
            # Fallback for models without chat templates
            system = system_prompt_override if system_prompt_override is not None else example["system_prompt"]
            question = example["question_template"].format(**example['metadata'])
            answer = example["assistant_template"].format(**example['metadata']) + eos_token
            prompt_str = f"{system}\n\nuser: {question}\n\nassistant: {answer}"
    else:
        # For evaluation, only include the question (no answer)
        if hasattr(tok, 'apply_chat_template'):
            prompt = [  
                {"role": "system", "content": system_prompt_override if system_prompt_override is not None else example.get("system_prompt", "")},
                {"role": "user", "content": example["question_template"].format(**example['metadata'])}
            ]
            prompt_str = tok.apply_chat_template(prompt, add_generation_prompt=True, 
                                                tokenize=False, enable_thinking=False)
        else:
            system = system_prompt_override if system_prompt_override is not None else example.get("system_prompt", "")
            question = example["question_template"].format(**example['metadata'])
            prompt_str = f"{system}\n\nuser: {question}\n\nassistant: "
    
    # Tokenize the prompt
    tokenized_output = tok(prompt_str, padding="max_length", truncation=True, max_length=max_len, return_tensors="np")
    
    # Ensure input_ids and attention_mask are 1D
    # The tokenizer returns (1, seq_len) for single examples, so we take the first element.
    input_ids = tokenized_output["input_ids"][0]
    attention_mask = tokenized_output["attention_mask"][0]
    
    processed_example = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
    if is_train:
        # For training: find the answer span in the prompt
        answer_text = str(example["assistant_template"].format(**example['metadata']) + eos_token)
        
        input_ids_list = input_ids.tolist() # Already 1D
        
        # Use robust helper to find answer token positions
        answer_span = find_answer_token_positions(tok, prompt_str, answer_text, input_ids_list, max_len)
        label = [-100] * len(input_ids_list)
        if answer_span is not None:
            start_idx, end_idx = answer_span
            for j in range(start_idx, end_idx):
                label[j] = input_ids_list[j]
        else:
            print(f"Warning: Answer not found in input_ids for example with answer {answer_text}")
        processed_example["labels"] = label  # Assign 1D list directly
        
        # Add position weights if requested
        if create_position_weights:
            position_weights = [default_weight] * len(input_ids_list)
            
            # Find positions within <answer> tags in the answer text
            import re
            for match in re.finditer(r'<answer>(.*?)</answer>', answer_text, re.DOTALL):
                tagged_content = match.group(1)  # Content inside the tags
                
                # Use the existing find_answer_token_positions function
                tag_token_span = find_answer_token_positions(tok, prompt_str, tagged_content, input_ids_list, max_len)
                if tag_token_span is not None:
                    tag_start_idx, tag_end_idx = tag_token_span
                    for k in range(tag_start_idx, tag_end_idx):
                        if k < len(position_weights):
                            position_weights[k] = answer_weight
            
            processed_example["position_weights"] = position_weights
    else:
        # For evaluation: right-align answer tokens. right aligned is conventionally used
        # for answers as sometimes an answer can have a -100 in the middle of it, so when
        # you scan from the left on right aligned data, you will find the answer including the 
        # internal -100s.
        formatted_answer = example["assistant_template"].format(**example['metadata']) + eos_token
        ans_enc = tok(formatted_answer, truncation=True, add_special_tokens=False, max_length=max_label_len, return_tensors="np")
        answer = ans_enc["input_ids"].tolist()[0]
        
        label = [-100] * max_len
        label[-len(answer):] = answer[-max_len:]
        processed_example["labels"] = label  # Assign 1D list directly
    
    return processed_example


# ─────────────────────────── metrics & helpers ────────────────────────

def find_answer_token_positions(tokenizer: Any, prompt_str: str, answer_str: str, input_ids_list: list, max_len: int) -> tuple | None:
    """
    Robustly find the token span in input_ids_list corresponding to answer_str in prompt_str.
    Uses offset mapping if available, otherwise falls back to best-effort substring search.
    Uses the same tokenizer options as main prompt tokenization: padding="max_length", truncation=True, max_length=max_len, return_tensors="np".

    :param tokenizer: The tokenizer object.
    :type tokenizer: Any
    :param prompt_str: The full prompt string.
    :type prompt_str: str
    :param answer_str: The answer string to locate.
    :type answer_str: str
    :param input_ids_list: The tokenized input_ids list for the prompt.
    :type input_ids_list: list
    :param max_len: The max length used for tokenization.
    :type max_len: int
    :return: (start_idx, end_idx) or None if not found.
    :rtype: tuple or None
    """
    # Try to get offset mapping
    try:
        enc = tokenizer(
            prompt_str,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="np",
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        offsets = enc.get("offset_mapping")[0]
        if offsets is not None:
            # Find answer substring in prompt
            answer_start = prompt_str.find(answer_str)
            if answer_start == -1:
                # Try to ignore whitespace differences
                match = re.search(re.escape(answer_str.strip()), prompt_str)
                if match:
                    answer_start = match.start()
                else:
                    return None
            answer_end = answer_start + len(answer_str)
            # Find token indices covering this span
            start_idx = end_idx = None
            for i, (s, e) in enumerate(offsets):
                if s <= answer_start < e:
                    start_idx = i
                if s < answer_end <= e:
                    end_idx = i+1
                    break
            if start_idx is not None and end_idx is not None:
                return (start_idx, end_idx)
            # Fallback: cover all tokens overlapping the span
            indices = [i for i, (s, e) in enumerate(offsets) if not (e <= answer_start or s >= answer_end)]
            if indices:
                return (indices[0], indices[-1]+1)
            return None
    except Exception as e:
        pass
    # Fallback: try to match answer token ids as a subsequence
    answer_enc = tokenizer(
        answer_str,
        add_special_tokens=False
    )
    answer_ids = answer_enc["input_ids"]
    
    # Ensure we have proper Python lists for comparison
    if hasattr(answer_ids, 'tolist'):
        answer_list = answer_ids.tolist()
    elif hasattr(answer_ids, '__iter__'):
        answer_list = list(answer_ids)
    else:
        answer_list = [answer_ids]
    
    for i in range(len(input_ids_list) - len(answer_list) + 1):
        # Convert input slice to list for comparison
        input_slice = input_ids_list[i : i + len(answer_list)]
        if hasattr(input_slice, 'tolist') and not isinstance(input_slice, list):
            input_slice = input_slice.tolist()
        elif not isinstance(input_slice, list):
            input_slice = list(input_slice)
        
        if input_slice == answer_list:
            return (i, i + len(answer_list))
    return None

def _norm(s: str, tokenizer: Any = None) -> str:
    """
    Normalize prediction/label strings for exact-match comparison.
    Extracts the answer after the last ": " until whitespace, period, or EOS token.
    If no colon, extracts everything after the last whitespace.

    :param s: Input string.
    :param tokenizer: Tokenizer instance to get the EOS token from.
    :return: Normalized string.
    """
    import re
    
    # Get EOS token from tokenizer if available
    eos_token = None
    if tokenizer is not None and hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
        eos_token = tokenizer.eos_token
    
    # Find the last colon followed by a space
    idx = s.rfind(': ')
    if idx != -1:
        # Extract everything after ": "
        answer_part = s[idx + 2:]
        
        # Handle different cases based on whether we have an EOS token
        if eos_token:
            # Split by EOS token and take the first part
            if eos_token in answer_part:
                answer_part = answer_part.split(eos_token)[0]
        
        # Now extract the answer until whitespace or period
        match = re.match(r"([^\s\.]*)", answer_part)
        if match:
            result = match.group(1).strip()
            return result
        return answer_part.strip()
    
    # If no ": " found, try to extract everything after the last whitespace
    # First clean up EOS tokens
    result = s.strip()
    if eos_token and result.endswith(eos_token):
        result = result[:-len(eos_token)].strip()
    
    # Find the last whitespace and extract everything after it
    last_space_idx = result.rfind(' ')
    if last_space_idx != -1:
        # Extract everything after the last space
        final_answer = result[last_space_idx + 1:].strip().rstrip('.')
        return final_answer
    
    # If no whitespace, clean the whole string
    return result.rstrip('.')


def _norm_tagged(s: str, tokenizer: Any = None) -> str:
    """
    Normalize prediction/label strings for exact-match comparison.
    Extracts the answer between <answer> and </answer> tags.
    If no tags found, returns the original string stripped.

    :param s: Input string.
    :param tokenizer: Tokenizer instance (unused, kept for compatibility).
    :return: Normalized string.
    """
    # Find the last occurrence of <answer> tag
    last_answer_start = s.rfind('<answer>')
    if last_answer_start != -1:
        # Find the corresponding </answer> tag after the last <answer>
        answer_end = s.find('</answer>', last_answer_start)
        if answer_end != -1:
            # Extract content between the tags
            start_pos = last_answer_start + len('<answer>')
            answer_content = s[start_pos:answer_end].strip()
            # Remove trailing period if present
            if answer_content.endswith('.'):
                answer_content = answer_content[:-1].strip()
            return answer_content
    
    # If no tags found, return the original string stripped
    return s.strip()


def compute_metrics_closure(tokenizer: Any, compare: str) -> Callable[[Any], Any]:
    """
    Compute metrics closure.

    :param tokenizer: Tokenizer instance.
    :param compare: Comparison type.
    :return: Metrics computation function.
    """
    all_preds = []
    all_labels = []

    def compute_metrics(eval_preds, compute_result: bool = True) -> dict:
        """
        Compute metrics. With batch_eval_metrics=True, this function is called per batch and at the end with compute_result=True.
        Accumulates predictions and labels across batches, and only computes/returns metrics when compute_result=True.

        :param eval_preds: Evaluation predictions.
        :param compute_result: Whether to return summary statistics (True at end of eval loop).
        :return: Computed metrics (only when compute_result=True).
        """
        nonlocal all_preds, all_labels
        preds, labels = eval_preds
        # Move to cpu and convert to numpy if needed
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if isinstance(preds, tuple):
            preds = preds[0]

        try:
            def filter_out_of_range(tokens, vocab_size, pad_token_id):
                # Replace out-of-range tokens with pad_token_id
                return [t if 0 <= t < vocab_size else pad_token_id for t in tokens]
            # Filter out-of-range tokens for preds and labels
            filtered_preds = [filter_out_of_range(seq, tokenizer.vocab_size, tokenizer.pad_token_id) for seq in preds]
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            filtered_labels = [filter_out_of_range(seq, tokenizer.vocab_size, tokenizer.pad_token_id) for seq in labels]
            decoded_preds = tokenizer.batch_decode(filtered_preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(filtered_labels, skip_special_tokens=True)
            decoded_preds = [_norm_tagged(p, tokenizer) for p in decoded_preds]
            decoded_labels = [_norm_tagged(label, tokenizer) for label in decoded_labels]
            all_preds.extend(decoded_preds)
            all_labels.extend(decoded_labels)
        except OverflowError:
            print(f"OverflowError: {preds}")
        
        if compute_result:
            # Compute metrics only on the final call
            # Custom exact match computation
            matches = 0
            total = len(all_preds)
            
            for pred, label in zip(all_preds, all_labels):
                is_match = False
                
                if label.startswith('[') or label.startswith('{'):
                    # Handle Python literals
                    try:
                        import ast
                        label_obj = ast.literal_eval(label)
                        pred_obj = ast.literal_eval(pred)
                        is_match = (label_obj == pred_obj)
                    except (ValueError, SyntaxError, TypeError):
                        # If literal evaluation fails, no match
                        is_match = False
                else:
                    # Handle regular string comparison
                    is_match = (pred == label)
                
                if is_match:
                    matches += 1
            
            exact_match_score = matches / total if total > 0 else 0.0
            exact_m = {"exact_match": exact_match_score}
            
            # Reset for next eval
            all_preds = []
            all_labels = []
            return exact_m if exact_m is not None else {}
        else:
            # Return empty dict on intermediate calls
            return {}
    return compute_metrics
