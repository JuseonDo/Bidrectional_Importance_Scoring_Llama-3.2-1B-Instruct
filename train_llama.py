# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Train Llama-3.2-1B for token importance scoring (bidirectional attention).
Removes causal mask and adds binary classification head.
"""

import argparse
import os
import random
import time
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
)
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.cache_utils import DynamicCache


MAX_GRAD_NORM = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Llama for token importance scoring (bidirectional)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Pretrained model name",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./checkpoints/llama_3.2_1b_meetingbank",
        help="Save path for model",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epoch", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--save_top_k",
        type=int,
        default=3,
        help="Save top k checkpoints based on F1 score",
    )
    parser.add_argument(
        "--val_every_n_steps",
        type=int,
        default=None,
        help="Run validation every N steps (overrides epoch-based validation)",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=None,
        help="Save checkpoint every N steps (overrides epoch-based saving)",
    )
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BidirectionalLlamaModel(LlamaModel):
    """
    Llama model with bidirectional attention (no causal mask).
    Overrides forward to skip causal mask creation.
    """

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_len = inputs_embeds.shape[:2]
        device = inputs_embeds.device

        # Create bidirectional attention mask (no causal masking)
        if attention_mask is not None:
            # Convert 2D attention mask to 4D
            # Shape: (batch_size, 1, seq_len, seq_len)
            extended_attention_mask = attention_mask[:, None, None, :].expand(
                batch_size, 1, seq_len, seq_len
            )
            # Convert to additive mask (0 -> 0.0, 1 -> -inf for masked positions)
            extended_attention_mask = (1.0 - extended_attention_mask.float()) * torch.finfo(inputs_embeds.dtype).min
            extended_attention_mask = extended_attention_mask.to(inputs_embeds.dtype)
        else:
            extended_attention_mask = None

        if cache_position is None:
            cache_position = torch.arange(seq_len, device=device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        from transformers.modeling_outputs import BaseModelOutputWithPast
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
        )


class LlamaForTokenImportance(LlamaPreTrainedModel):
    """
    Llama model with bidirectional attention for token importance scoring.
    Binary classification head: 0 = prune, 1 = keep.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.model = BidirectionalLlamaModel(config)

        classifier_dropout = getattr(config, "classifier_dropout", 0.1)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> TokenClassifierOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            **kwargs,
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def align_and_create_labels(
    original_text: str,
    compressed_text: str,
    tokenizer,
) -> Tuple[List[int], List[int]]:
    """
    Align original and compressed text to create binary labels.
    Returns: (token_ids, labels) where label=1 if token is kept, 0 if pruned.
    """
    # Tokenize both texts
    original_tokens = tokenizer.tokenize(original_text)
    compressed_tokens = tokenizer.tokenize(compressed_text)

    # Create a set of compressed tokens for fast lookup
    compressed_set = set(compressed_tokens)

    # Simple alignment: mark token as kept if it appears in compressed version
    # Use sliding window approach for better alignment
    labels = []
    compressed_idx = 0

    for orig_token in original_tokens:
        # Check if this token matches the next expected compressed token
        if compressed_idx < len(compressed_tokens):
            if orig_token == compressed_tokens[compressed_idx]:
                labels.append(1)  # Keep
                compressed_idx += 1
            else:
                # Check if token exists later in compressed (might be reordered)
                found = False
                for look_ahead in range(min(5, len(compressed_tokens) - compressed_idx)):
                    if orig_token == compressed_tokens[compressed_idx + look_ahead]:
                        # Found it, mark intermediate ones as dropped
                        labels.append(1)
                        compressed_idx += look_ahead + 1
                        found = True
                        break
                if not found:
                    labels.append(0)  # Prune
        else:
            labels.append(0)  # Prune - no more compressed tokens

    token_ids = tokenizer.convert_tokens_to_ids(original_tokens)
    return token_ids, labels


class MeetingBankDataset(Dataset):
    """
    Dataset for MeetingBank-LLMCompressed.
    Uses prompt_list and compressed_prompt_list directly.
    Creates binary labels based on token survival in compressed version.
    """

    def __init__(
        self,
        data,
        tokenizer,
    ):
        self.tokenizer = tokenizer

        # Flatten all prompts
        self.samples = []
        for item in data:
            prompt_list = item["prompt_list"]
            compressed_list = item["compressed_prompt_list"]

            # Use each prompt individually
            for original, compressed in zip(prompt_list, compressed_list):
                self.samples.append((original, compressed))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        original, compressed = self.samples[idx]

        # Tokenize original text (no padding here, will pad in collate_fn)
        original_encoding = self.tokenizer(
            original,
            return_tensors="pt",
        )

        input_ids = original_encoding["input_ids"].squeeze(0)
        attention_mask = original_encoding["attention_mask"].squeeze(0)

        # Create labels by aligning with compressed text
        labels = self._create_labels(original, compressed, input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _create_labels(
        self, original: str, compressed: str, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Create binary labels for each token.
        1 = keep (token appears in compressed version)
        0 = prune (token is removed in compressed version)
        -100 = ignore (special tokens)
        """
        # Get tokens from input_ids
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())

        # Tokenize compressed text
        compressed_tokens = self.tokenizer.tokenize(compressed)

        # Create labels
        labels = []
        compressed_idx = 0

        for i, token in enumerate(tokens):
            if token in [self.tokenizer.bos_token, self.tokenizer.eos_token]:
                labels.append(-100)  # Ignore special tokens
            else:
                # Try to match with compressed tokens
                if compressed_idx < len(compressed_tokens):
                    # Normalize tokens for comparison
                    orig_clean = token.replace("▁", "").replace("Ġ", "").lower()
                    comp_clean = compressed_tokens[compressed_idx].replace("▁", "").replace("Ġ", "").lower()

                    if orig_clean == comp_clean or token == compressed_tokens[compressed_idx]:
                        labels.append(1)  # Keep
                        compressed_idx += 1
                    else:
                        # Look ahead to find match
                        found = False
                        for look_ahead in range(min(10, len(compressed_tokens) - compressed_idx)):
                            comp_ahead = compressed_tokens[compressed_idx + look_ahead]
                            comp_ahead_clean = comp_ahead.replace("▁", "").replace("Ġ", "").lower()
                            if orig_clean == comp_ahead_clean or token == comp_ahead:
                                labels.append(1)
                                compressed_idx = compressed_idx + look_ahead + 1
                                found = True
                                break
                        if not found:
                            labels.append(0)  # Prune
                else:
                    labels.append(0)  # Prune - no more compressed tokens

        return torch.tensor(labels, dtype=torch.long)


def collate_fn(batch, pad_token_id):
    """
    Collate function for dynamic padding within batch.
    Pads to the longest sequence in the batch.
    """
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        # Pad input_ids with pad_token_id
        input_ids = F.pad(item["input_ids"], (0, pad_len), value=pad_token_id)
        input_ids_list.append(input_ids)

        # Pad attention_mask with 0
        attention_mask = F.pad(item["attention_mask"], (0, pad_len), value=0)
        attention_mask_list.append(attention_mask)

        # Pad labels with -100 (ignore index)
        labels = F.pad(item["labels"], (0, pad_len), value=-100)
        labels_list.append(labels)

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
    }


def train_epoch(
    model,
    train_dataloader,
    optimizer,
    device,
    epoch,
    writer,
    gradient_accumulation_steps,
    global_step=0,
    val_every_n_steps=None,
    save_every_n_steps=None,
    val_dataloader=None,
    save_path=None,
    tokenizer=None,
    top_k_checkpoints=None,
    save_top_k=3,
):
    model.train()
    tr_loss = 0
    tr_accuracy = 0
    nb_tr_steps = 0
    tr_preds, tr_labels = [], []

    optimizer.zero_grad()

    pbar = tqdm(train_dataloader, desc=f"Training Epoch {epoch}")
    for idx, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        tr_loss += outputs.loss.item()
        nb_tr_steps += 1

        # Calculate accuracy on non-ignored tokens
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        mask = labels != -100
        if mask.sum() > 0:
            valid_preds = predictions[mask]
            valid_labels = labels[mask]
            accuracy = (valid_preds == valid_labels).float().mean().item()
            tr_accuracy += accuracy

            tr_preds.extend(valid_preds.cpu().tolist())
            tr_labels.extend(valid_labels.cpu().tolist())

        if (idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            # Step-based validation and saving
            if val_every_n_steps and global_step % val_every_n_steps == 0:
                print(f"\n[Step {global_step}] Running validation...")
                acc, f1 = evaluate(model, val_dataloader, device, global_step, writer, save_path=save_path, tokenizer=tokenizer, step_based=True)
                model.train()  # Switch back to training mode

                # Step-based checkpoint saving
                if save_every_n_steps and global_step % save_every_n_steps == 0:
                    _save_checkpoint(model, tokenizer, save_path, global_step, f1, acc, top_k_checkpoints, save_top_k, step_based=True)

        if idx % 100 == 0:
            avg_loss = tr_loss / nb_tr_steps
            avg_acc = tr_accuracy / nb_tr_steps
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.4f}", "step": global_step})
            writer.add_scalar("Loss/train", avg_loss, global_step)
            writer.add_scalar("Acc/train", avg_acc, global_step)

    # Handle remaining gradients
    if len(train_dataloader) % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

    avg_loss = tr_loss / nb_tr_steps
    avg_acc = tr_accuracy / nb_tr_steps

    # Calculate F1 score
    if tr_labels:
        f1 = f1_score(tr_labels, tr_preds, average="binary")
    else:
        f1 = 0.0

    print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, F1: {f1:.4f}")
    return avg_loss, avg_acc, global_step


def _save_checkpoint(model, tokenizer, save_path, step_or_epoch, f1, acc, top_k_checkpoints, save_top_k, step_based=False):
    """Helper function to save checkpoints with top-k logic."""
    import heapq
    import shutil

    if step_based:
        ckpt_path = os.path.join(save_path, f"step_{step_or_epoch}_f1_{f1:.4f}")
    else:
        ckpt_path = os.path.join(save_path, f"epoch_{step_or_epoch}_f1_{f1:.4f}")

    # Save last checkpoint
    last_path = os.path.join(save_path, "last")
    os.makedirs(last_path, exist_ok=True)
    model.save_pretrained(last_path)
    tokenizer.save_pretrained(last_path)
    torch.save({
        "step" if step_based else "epoch": step_or_epoch,
        "f1": f1,
        "acc": acc,
    }, os.path.join(last_path, "training_state.pt"))

    # Save top k checkpoints
    if len(top_k_checkpoints) < save_top_k:
        os.makedirs(ckpt_path, exist_ok=True)
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        heapq.heappush(top_k_checkpoints, (f1, step_or_epoch, ckpt_path))
        print(f"Saved checkpoint: {ckpt_path}")
    elif f1 > top_k_checkpoints[0][0]:
        _, _, worst_path = heapq.heappop(top_k_checkpoints)
        if os.path.exists(worst_path):
            shutil.rmtree(worst_path)
            print(f"Removed checkpoint: {worst_path}")

        os.makedirs(ckpt_path, exist_ok=True)
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        heapq.heappush(top_k_checkpoints, (f1, step_or_epoch, ckpt_path))
        print(f"Saved checkpoint: {ckpt_path}")


def evaluate(model, eval_dataloader, device, epoch, writer, save_path=None, tokenizer=None, step_based=False):
    model.eval()
    eval_loss = 0
    eval_accuracy = 0
    nb_eval_steps = 0
    eval_preds, eval_labels_list = [], []

    # Store per-instance results
    instance_results = []
    instance_idx = 0

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            eval_loss += outputs.loss.item()
            nb_eval_steps += 1

            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)  # [batch, seq_len, 2]
            predictions = torch.argmax(logits, dim=-1)

            # Process each instance in batch
            batch_size = input_ids.size(0)
            for i in range(batch_size):
                inst_input_ids = input_ids[i]
                inst_labels = labels[i]
                inst_preds = predictions[i]
                inst_probs = probs[i, :, 1]  # prob of class 1 (keep)
                inst_mask = inst_labels != -100

                if inst_mask.sum() > 0:
                    valid_preds = inst_preds[inst_mask].cpu().tolist()
                    valid_labels = inst_labels[inst_mask].cpu().tolist()
                    valid_probs = inst_probs[inst_mask].cpu().tolist()

                    # Calculate per-instance metrics
                    inst_acc = sum(p == l for p, l in zip(valid_preds, valid_labels)) / len(valid_labels)
                    inst_f1 = f1_score(valid_labels, valid_preds, average="binary", zero_division=0)

                    # Decode tokens if tokenizer provided
                    if tokenizer:
                        tokens = tokenizer.convert_ids_to_tokens(inst_input_ids[inst_mask].cpu().tolist())
                    else:
                        tokens = None

                    instance_results.append({
                        "instance_idx": instance_idx,
                        "tokens": tokens,
                        "labels": valid_labels,
                        "predictions": valid_preds,
                        "keep_probs": valid_probs,
                        "accuracy": inst_acc,
                        "f1": inst_f1,
                    })

                instance_idx += 1

            mask = labels != -100
            if mask.sum() > 0:
                valid_preds = predictions[mask]
                valid_labels = labels[mask]
                accuracy = (valid_preds == valid_labels).float().mean().item()
                eval_accuracy += accuracy

                eval_preds.extend(valid_preds.cpu().tolist())
                eval_labels_list.extend(valid_labels.cpu().tolist())

    avg_loss = eval_loss / nb_eval_steps
    avg_acc = eval_accuracy / nb_eval_steps

    # Calculate F1 score
    if eval_labels_list:
        f1 = f1_score(eval_labels_list, eval_preds, average="binary")
    else:
        f1 = 0.0

    print(f"Eval Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, F1: {f1:.4f}")

    writer.add_scalar("Loss/eval", avg_loss, epoch)
    writer.add_scalar("Acc/eval", avg_acc, epoch)
    writer.add_scalar("F1/eval", f1, epoch)

    # Save per-instance results to JSON in eval_results folder
    if save_path:
        import json
        eval_dir = os.path.join(save_path, "eval_results")
        os.makedirs(eval_dir, exist_ok=True)

        eval_results = {
            "step" if step_based else "epoch": epoch,
            "avg_loss": avg_loss,
            "avg_accuracy": avg_acc,
            "avg_f1": f1,
            "num_instances": len(instance_results),
            "instances": instance_results,
        }
        prefix = "eval_step" if step_based else "eval_epoch"
        eval_json_path = os.path.join(eval_dir, f"{prefix}_{epoch}.json")
        with open(eval_json_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"Saved evaluation results to {eval_json_path}")

    return avg_acc, f1


def main():
    args = parse_args()
    seed_everything(args.seed)

    # Add timestamp to save_path and log_dir
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{args.save_path}_{timestamp}"
    log_dir = save_path.replace("checkpoints", "logs")

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Save source code to checkpoint directory
    import shutil
    import json
    source_file = os.path.abspath(__file__)
    shutil.copy(source_file, os.path.join(save_path, "train_llama.py"))
    print(f"Saved source code to {save_path}/train_llama.py")

    # Save all hyperparameters (including defaults)
    hparams = vars(args).copy()
    hparams["timestamp"] = timestamp
    hparams["save_path_with_timestamp"] = save_path
    hparams["log_dir"] = log_dir
    with open(os.path.join(save_path, "hparams.json"), "w") as f:
        json.dump(hparams, f, indent=2)
    print(f"Saved hyperparameters to {save_path}/hparams.json")

    print(f"Checkpoint path: {save_path}")
    print(f"Log path: {log_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer and model
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model_name)
    config.classifier_dropout = 0.1

    model = LlamaForTokenImportance.from_pretrained(
        args.model_name,
        config=config,
        dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # Load dataset
    print("Loading MeetingBank-LLMCompressed dataset...")
    dataset = load_dataset("microsoft/MeetingBank-LLMCompressed", split="train")

    # Split into train/val (8:2)
    dataset = dataset.shuffle(seed=args.seed)
    split_idx = int(len(dataset) * 0.8)
    train_data = dataset.select(range(split_idx))
    val_data = dataset.select(range(split_idx, len(dataset)))

    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    # Create datasets
    train_dataset = MeetingBankDataset(train_data, tokenizer)
    val_dataset = MeetingBankDataset(val_data, tokenizer)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Print sample
    sample = train_dataset[0]
    print("\n=== Sample ===")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print(f"Non-padding tokens: {(sample['attention_mask'] == 1).sum().item()}")
    print(f"Keep tokens: {(sample['labels'] == 1).sum().item()}")
    print(f"Prune tokens: {(sample['labels'] == 0).sum().item()}")

    # Create dataloaders with dynamic padding (use eos_token for padding)
    from functools import partial
    collate = partial(collate_fn, pad_token_id=tokenizer.eos_token_id)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Training loop
    import heapq
    import shutil

    # Min-heap to track top k checkpoints: (f1, step_or_epoch)
    top_k_checkpoints = []
    global_step = 0

    # Check if using step-based validation/saving
    step_based = args.val_every_n_steps is not None or args.save_every_n_steps is not None
    if step_based:
        print(f"Using step-based validation every {args.val_every_n_steps} steps")
        print(f"Using step-based saving every {args.save_every_n_steps} steps")

    for epoch in range(args.num_epoch):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.num_epoch}")
        print(f"{'='*50}")

        _, _, global_step = train_epoch(
            model,
            train_dataloader,
            optimizer,
            device,
            epoch,
            writer,
            args.gradient_accumulation_steps,
            global_step=global_step,
            val_every_n_steps=args.val_every_n_steps,
            save_every_n_steps=args.save_every_n_steps,
            val_dataloader=val_dataloader,
            save_path=save_path,
            tokenizer=tokenizer,
            top_k_checkpoints=top_k_checkpoints,
            save_top_k=args.save_top_k,
        )

        # Epoch-based validation and saving (only if not using step-based)
        if not step_based:
            acc, f1 = evaluate(model, val_dataloader, device, epoch, writer, save_path=save_path, tokenizer=tokenizer)
            _save_checkpoint(model, tokenizer, save_path, epoch, f1, acc, top_k_checkpoints, args.save_top_k, step_based=False)

    # Print final results
    best_f1 = max(f1 for f1, _, _ in top_k_checkpoints) if top_k_checkpoints else 0
    print(f"\nTraining complete. Best F1: {best_f1:.4f}")
    print(f"Top {args.save_top_k} checkpoints:")
    for f1, step_or_epoch, path in sorted(top_k_checkpoints, reverse=True):
        label = "Step" if step_based else "Epoch"
        print(f"  {label} {step_or_epoch}: F1={f1:.4f} -> {path}")
    writer.close()


if __name__ == "__main__":
    main()
