from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
import numpy as np
import torch
import re

EPSILON=1e-15
def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded

@dataclass
class SFTCausalCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 8
    label_pad_token_id: int = -100
    return_tensors: str = "pt"


    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        if labels is not None and all(label is None for label in labels):
            labels = None
        non_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]

        # run through tokenizer without labels to ensure no side effects
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        no_padding = self.padding is False or self.padding == PaddingStrategy.DO_NOT_PAD

        if labels is not None:
            if no_padding:
                if isinstance(features[0][label_name], list):
                    batch["labels"] = list(labels)
                else:
                    batch["labels"] = [np.concatenate([label, []]) for label in labels]
            else:
                max_padding = self.padding == PaddingStrategy.MAX_LENGTH and self.max_length is not None
                max_label_length = max(len(l) for l in labels) if not max_padding else self.max_length
                if self.pad_to_multiple_of is not None and (max_label_length % self.pad_to_multiple_of != 0):
                    # max_label_length = (
                    #     (max_label_length + self.pad_to_multiple_of - 1)
                    #     // self.pad_to_multiple_of
                    #     * self.pad_to_multiple_of
                    # )
                    max_label_length = ((max_label_length // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

                padding_side = self.tokenizer.padding_side
                if isinstance(features[0][label_name], list):
                    batch["labels"] = [
                        label + [self.label_pad_token_id] * (max_label_length - len(label))
                        if padding_side == "right"
                        else [self.label_pad_token_id] * (max_label_length - len(label)) + label
                        for label in labels
                    ]
                else:
                    batch["labels"] = [
                        np.concatenate(
                            [
                                label,
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                            ]
                        )
                        if padding_side == "right"
                        else np.concatenate(
                            [
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                                label,
                            ]
                        )
                        for label in labels
                    ]

        # reintroduce side effects via tokenizer that return respective datatypes for the `return_tensors` argument
        if batch.get("labels", None) is not None:
            if self.return_tensors == "pt":
                import torch

                batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
            elif self.return_tensors == "tf":
                import tensorflow as tf

                batch["labels"] = tf.constant(batch["labels"], dtype=tf.int64)
            else:
                batch["labels"] = np.array(batch["labels"], dtype=np.int64)
        else:
            batch["labels"] = None

        
        return batch

def entropy(logits, is_logits=True, temperature=1.0):
    """
    calculate the entropy of the last dimension of given logits

    input:
    logits: torch.Tensor. 
    temperature: float, the temperature used in softmax

    return:
    torch.Tensor
    """
    
    if is_logits:
        probs = torch.nn.functional.softmax(logits/float(temperature), dim=-1)
    
    
    return (-torch.log(torch.clamp(probs, min=EPSILON)) * probs).sum(dim=-1)

def loss_function(logits, labels, loss_name = "CE", ignore_index=-100, topk=3, logsoftmax=True, entropy_weight=True, token_imps=None):
    """
    logits : batch_size x sequence_length x vocab_size
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous() # batch_size x (sequence_length - 1)
    if token_imps is not None:
        
        shift_token_imps = token_imps[..., 1:].contiguous()  # batch_size x (sequence_length - 1)
    if entropy_weight:
        ent = entropy(shift_logits.detach().clone(), is_logits=True, temperature=1.0)
    if loss_name == "CE":
        if token_imps is not None:
            shift_logprobs = shift_logits.log_softmax(dim=-1) * shift_token_imps.unsqueeze(-1)
                
            loss_fct = torch.nn.NLLLoss(ignore_index=ignore_index, reduction="mean")
            loss = loss_fct(shift_logprobs.view(-1, shift_logprobs.shape[-1]), shift_labels.view(-1))

            mask = (shift_labels != ignore_index).float()
            active_counts = mask.sum(dim=1)
            masked_ent = ent *  mask
            
            mean_ent = masked_ent.sum(dim=1)/ active_counts
        else:
        
            if entropy_weight:

                shift_logprobs = shift_logits.log_softmax(dim=-1) * ent.unsqueeze(-1)

                loss_fct = torch.nn.NLLLoss(ignore_index=ignore_index, reduction="mean")
                loss = loss_fct(shift_logprobs.view(-1, shift_logprobs.shape[-1]), shift_labels.view(-1))

                mask = (shift_labels != ignore_index).float()
                active_counts = mask.sum(dim=1)
                masked_ent = ent *  mask
                mean_ent = masked_ent.sum(dim=1)/ active_counts
            else:
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

                loss = loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
                mean_ent = None
        
    elif loss_name == "topk":
        if logsoftmax:
            shift_logprobs = shift_logits.log_softmax(dim=-1)
        else:
            shift_logprobs = shift_logits
        topk_logprobs, _ = torch.topk(shift_logprobs, topk, dim=-1) # batch_size x seq_len x topk
        if entropy_weight:
            shift_logprobs = (topk_logprobs.contiguous() * ent.unsqueeze(-1)) .view(-1, topk)
        
            active_logprobs = shift_logprobs[shift_labels.view(-1) != ignore_index]

            mask = (shift_labels != ignore_index).float()
            active_counts = mask.sum(dim=1)
            masked_ent = ent *  mask
            mean_ent = masked_ent.sum(dim=1)/ active_counts
        else:
            shift_logprobs = topk_logprobs.contiguous().view(-1, topk)
        
            active_logprobs = shift_logprobs[shift_labels.view(-1) != ignore_index]
            mean_ent = None
        loss = -active_logprobs.mean()
        
    return loss, mean_ent


def grad_vector_and_weight(model, layer_num, verbose=False):
    lmd = 0.3
    names = []
    grads = []
    weights = []
    last_layer_grads = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            names.append(name)
            grads.append(p.grad.data.view(-1))
        
        
            if "layer" in name and "weight" in name:
                nums = re.findall(r"\d+", name)
                
                if len(nums) > 0:
                    w = np.exp((int(nums[0])+1)*lmd)
                    weights.append(w)
                    if int(nums[0]) == layer_num -2:
                        last_layer_grads.append(p.grad.data.view(-1))
                else:
                    raise ValueError(f"param name '{name}' do not contain layer num information")
            elif "lm_head" in name:
                w = np.exp(layer_num*lmd)
                weights.append(w)
            else:
                if verbose:
                    print(f"{name} is not a layer param nor lm_head param, weight set to zero")
                weights.append(0)
        else:
            print(f"param 'name' have none grad.")
    
    # grads = [g.cpu() for g in grads]
    # last_laye, r_grads = [g.cpu() for g in last_layer_grads]
    
    return names, grads, weights, last_layer_grads

def grad_norms(param_grads, device=None, p=1, mean=True):
    """
    Compute mean L1 and L2 norm of gradients without concatenation.

    Args:
        param_grads: iterable of tensors (grads of each layer).
                     Some may be None.
        device: torch device where accumulation happens (default: same as first grad).

    Returns:
        mean_l1: scalar (float)
        mean_l2: scalar (float)
    """
    # Pick device from first non-None grad if not given
    if device is None:
        for g in param_grads:
            if g is not None:
                device = g.device
                break
        else:
            raise ValueError("No gradients provided")
    if p==1:
        total_abs = torch.zeros((), device=device)  # sum of |g|
    elif p==2:
        total_sq  = torch.zeros((), device=device)  # sum of g^2
    total_num = 0

    with torch.no_grad():
        for g in param_grads:
            if g is None:
                continue
            if p==1:
                total_abs += g.abs().sum()
            elif p==2:
                total_sq  += (g**2).sum()
            total_num += g.numel()
    if p == 1:
        if mean:
            return (total_abs / total_num)
        else:
            return total_abs
    elif p == 2:
        if mean:
            return torch.sqrt(total_sq)/np.sqrt(total_num)
        else:
            return torch.sqrt(total_sq)
        