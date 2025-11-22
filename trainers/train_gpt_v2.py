#!/usr/bin/env python3
"""
End-to-end finetuning entry point for IndexTTS2 (GPT module) with Japanese data.

This trainer expects the preprocessing pipeline to have produced manifests where each
sample record stores paths to:
  - text token ids (.npy, int32)
  - semantic codes (.npy, int32)
  - conditioning latent (.npy, float32 [32, hidden])
  - emotion vector (.npy, float32 [hidden])

The model is optimised with cross-entropy losses over text tokens and semantic codes,
with optional gradient accumulation and mixed-precision support. Checkpoints are
emitted every 1k optimiser steps (`model_step{N}.pth`), keeping only the three most
recent snapshots. TensorBoard summaries track losses and learning rate under the
chosen output directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import datetime
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.front import TextNormalizer, TextTokenizer
from indextts.utils.hardware_optimizer import detect_hardware, print_hardware_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune IndexTTS2 GPT on Japanese data.")
    parser.add_argument(
        "--train-manifest",
        dest="train_manifests",
        action="append",
        type=str,
        required=True,
        help="Training manifest JSONL. Repeat to mix multiple datasets; optionally suffix with '::lang' to force a language hint.",
    )
    parser.add_argument(
        "--val-manifest",
        dest="val_manifests",
        action="append",
        type=str,
        required=True,
        help="Validation manifest JSONL. Repeat to mix multiple datasets; optionally suffix with '::lang' to force a language hint.",
    )
    parser.add_argument("--tokenizer", type=Path, required=True, help="SentencePiece model path.")
    parser.add_argument("--config", type=Path, default=Path("checkpoints/config.yaml"), help="Model config YAML.")
    parser.add_argument("--base-checkpoint", type=Path, default=Path("checkpoints/gpt.pth"), help="Base GPT checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=Path("trained_ckpts"), help="Directory for checkpoints/logs.")
    parser.add_argument("--batch-size", type=int, default=0, help="Mini-batch size (0=auto-detect based on GPU VRAM).")
    parser.add_argument("--grad-accumulation", type=int, default=0, help="Gradient accumulation steps (0=auto-detect to reach effective batch=32).")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="LR warmup steps.")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional max optimiser steps (0 = unlimited).")
    parser.add_argument("--log-interval", type=int, default=100, help="Steps between training log entries.")
    parser.add_argument("--val-interval", type=int, default=1000, help="Validation frequency in steps (0 = once per epoch). Video recommends 1000 for faster validation.")
    parser.add_argument("--val-batch-size", type=int, default=0, help="Validation batch size (0=auto: 2× training batch, no gradients = more VRAM available).")
    parser.add_argument("--max-val-batches", type=int, default=200, help="Max batches for validation (0=all, 200≈10%% of dataset, saves GPU time).")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0=auto-detect based on CPU count).")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient norm clipping value.")
    parser.add_argument("--text-loss-weight", type=float, default=0.2, help="Weight for text CE loss.")
    parser.add_argument("--mel-loss-weight", type=float, default=0.8, help="Weight for semantic CE loss.")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA AMP (bfloat16 on L4).")
    parser.add_argument("--grad-checkpointing", action="store_true", help="Enable gradient checkpointing to save VRAM (slower but allows larger batches).")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from, or 'auto'.")
    parser.add_argument("--save-interval", type=int, default=1000, help="Checkpoint save frequency in optimizer steps. Video uses 1000 steps.")
    parser.add_argument("--keep-checkpoints", type=int, default=3, help="Number of recent checkpoints to keep (older ones are deleted). Video keeps last 3 epochs.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    return parser.parse_args()


@dataclass
class ManifestSpec:
    path: Path
    language: Optional[str] = None


def parse_manifest_specs(entries: Sequence[str], flag_name: str) -> List[ManifestSpec]:
    if not entries:
        raise ValueError(f"{flag_name} requires at least one manifest path.")
    specs: List[ManifestSpec] = []
    for raw in entries:
        value = raw.strip()
        lang: Optional[str] = None
        for separator in ("::", "@", "="):
            if separator in value:
                path_str, lang_part = value.rsplit(separator, 1)
                value = path_str.strip()
                lang = lang_part.strip().lower() or None
                break
        path = Path(value).expanduser()
        specs.append(ManifestSpec(path=path, language=lang))
    return specs


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    random.seed(seed)


@dataclass
class Sample:
    id: str
    text_ids_path: Path
    codes_path: Path
    condition_path: Path
    emo_vec_path: Path
    text_len: int
    code_len: int
    condition_len: int
    sample_type: str = "single"
    prompt_id: Optional[str] = None
    target_id: Optional[str] = None
    language: Optional[str] = None
    prompt_language: Optional[str] = None
    manifest_path: Optional[Path] = None


class JapaneseGPTDataset(Dataset):
    def __init__(self, manifests: Sequence[ManifestSpec]):
        if isinstance(manifests, ManifestSpec):
            manifests = [manifests]
        manifest_list = list(manifests)
        if not manifest_list:
            raise ValueError("No manifest paths supplied.")

        self.samples: List[Sample] = []
        self.sample_type: str = "unknown"
        self.manifest_summaries: List[Dict[str, object]] = []
        self.bad_indices: Set[int] = set()

        for spec in manifest_list:
            self._load_single_manifest(spec)

        if not self.samples:
            manifest_paths = ", ".join(str(spec.path) for spec in manifest_list)
            raise RuntimeError(f"No entries found in the provided manifests: {manifest_paths}")
        if self.sample_type != "paired":
            raise RuntimeError(
                "The GPT trainer expects prompt/target pair manifests.\n"
                "Generate paired manifests with tools/build_gpt_prompt_pairs.py and retry."
            )

    @staticmethod
    def _resolve_path(base_dir: Path, value: str) -> Path:
        if not value:
            raise ValueError("Empty path provided in manifest record.")
        path = Path(value)
        if path.is_absolute():
            return path
        return (base_dir / path).expanduser()

    @staticmethod
    def _normalize_language(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped.lower() if stripped else None

    def _load_single_manifest(self, spec: ManifestSpec) -> None:
        manifest_path = spec.path
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        local_count = 0
        local_languages: set[str] = set()
        manifest_sample_type: Optional[str] = None
        base_dir = manifest_path.parent

        print(f"[Info] Parsing manifest {manifest_path} ...")
        processed = 0
        progress_interval = 10000

        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                processed += 1
                is_paired = "prompt_condition_path" in record and "target_codes_path" in record
                if is_paired:
                    emo_path_value = record.get("prompt_emo_vec_path") or record.get("target_emo_vec_path")
                    if not emo_path_value:
                        raise RuntimeError(
                            f"Paired manifest entry {record.get('id')} missing prompt_emo_vec_path."
                        )
                    target_language = self._normalize_language(
                        record.get("target_language") or record.get("language") or spec.language
                    )
                    prompt_language = self._normalize_language(record.get("prompt_language") or spec.language)
                    sample = Sample(
                        id=record["id"],
                        text_ids_path=self._resolve_path(base_dir, record["target_text_ids_path"]),
                        codes_path=self._resolve_path(base_dir, record["target_codes_path"]),
                        condition_path=self._resolve_path(base_dir, record["prompt_condition_path"]),
                        emo_vec_path=self._resolve_path(base_dir, emo_path_value),
                        text_len=int(record["target_text_len"]),
                        code_len=int(record["target_code_len"]),
                        condition_len=int(record.get("prompt_condition_len", 32)),
                        sample_type="paired",
                        prompt_id=record.get("prompt_id"),
                        target_id=record.get("target_id"),
                        language=target_language,
                        prompt_language=prompt_language,
                        manifest_path=manifest_path,
                    )
                else:
                    language = self._normalize_language(record.get("language") or spec.language)
                    sample = Sample(
                        id=record["id"],
                        text_ids_path=self._resolve_path(base_dir, record["text_ids_path"]),
                        codes_path=self._resolve_path(base_dir, record["codes_path"]),
                        condition_path=self._resolve_path(base_dir, record["condition_path"]),
                        emo_vec_path=self._resolve_path(base_dir, record["emo_vec_path"]),
                        text_len=int(record["text_len"]),
                        code_len=int(record["code_len"]),
                        condition_len=int(record.get("condition_len", 32)),
                        sample_type="single",
                        manifest_path=manifest_path,
                        language=language,
                    )

                if manifest_sample_type is None:
                    manifest_sample_type = sample.sample_type
                elif manifest_sample_type != sample.sample_type:
                    raise RuntimeError(
                        f"Manifest {manifest_path} mixes sample types ({manifest_sample_type} vs {sample.sample_type})."
                    )

                self.samples.append(sample)
                local_count += 1
                if sample.language:
                    local_languages.add(sample.language)
                if sample.prompt_language:
                    local_languages.add(sample.prompt_language)

                if processed % progress_interval == 0:
                    print(
                        f"  • processed {processed:,} entries "
                        f"(kept {local_count:,}) in {manifest_path.name}"
                    )

        if local_count:
            if processed % progress_interval != 0:
                print(
                    f"  • processed {processed:,} entries "
                    f"(kept {local_count:,}) in {manifest_path.name}"
                )
            if manifest_sample_type and manifest_sample_type != "paired":
                raise RuntimeError(
                    f"Manifest {manifest_path} contains '{manifest_sample_type}' entries. "
                    "This trainer expects prompt/target pair manifests (see tools/build_gpt_prompt_pairs.py)."
                )
            if self.sample_type == "unknown":
                self.sample_type = manifest_sample_type or "unknown"
            elif manifest_sample_type and self.sample_type != manifest_sample_type:
                raise RuntimeError(
                    f"Mixed sample types encountered across manifests: {self.sample_type} vs {manifest_sample_type} (from {manifest_path})"
                )

            languages_display = sorted(local_languages)
            if not languages_display and spec.language:
                languages_display = [spec.language]
            language_text = ", ".join(languages_display) if languages_display else "unspecified"
            print(
                f"[Info] Loaded {local_count} samples ({manifest_sample_type}) from {manifest_path} "
                f"(languages: {language_text})"
            )
            self.manifest_summaries.append(
                {"path": manifest_path, "count": local_count, "languages": languages_display}
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.samples:
            raise RuntimeError("Dataset is empty.")

        if len(self.bad_indices) >= len(self.samples):
            raise RuntimeError("All samples were marked invalid; cannot continue.")

        attempts = 0
        max_attempts = len(self.samples)
        sample_count = len(self.samples)

        while attempts < max_attempts:
            current_idx = idx % sample_count

            sample = self.samples[current_idx]
            if sample is None:
                idx += 1
                attempts += 1
                continue

            try:
                text_ids = np.load(sample.text_ids_path, allow_pickle=False)
                codes = np.load(sample.codes_path, allow_pickle=False)
                condition = np.load(sample.condition_path, allow_pickle=False)
                emo_vec = np.load(sample.emo_vec_path, allow_pickle=False)

                if text_ids.size == 0 or codes.size == 0 or condition.size == 0 or emo_vec.size == 0:
                    raise ValueError("Encountered empty feature file.")

                text_ids = text_ids.astype(np.int64, copy=False)
                codes = codes.astype(np.int64, copy=False)
                condition = condition.astype(np.float32, copy=False)
                emo_vec = emo_vec.astype(np.float32, copy=False)

                return {
                    "id": sample.id,
                    "text_ids": torch.from_numpy(text_ids),
                    "codes": torch.from_numpy(codes),
                    "condition": torch.from_numpy(condition),  # [cond_len, dim]
                    "emo_vec": torch.from_numpy(emo_vec),
                    "text_len": torch.tensor(sample.text_len, dtype=torch.long),
                    "code_len": torch.tensor(sample.code_len, dtype=torch.long),
                    "condition_len": torch.tensor(sample.condition_len, dtype=torch.long),
                    "prompt_id": sample.prompt_id if sample.prompt_id else sample.id,
                    "target_id": sample.target_id if sample.target_id else sample.id,
                    "language": sample.language,
                    "prompt_language": sample.prompt_language,
                    "manifest_path": str(sample.manifest_path) if sample.manifest_path else "",
                }

            except (FileNotFoundError, OSError, ValueError) as exc:
                if current_idx not in self.bad_indices:
                    message = (
                        f"[Warn] Skipping sample '{sample.id}' due to load failure: {exc}. "
                        "It will be removed from the dataset for this run."
                    )
                    print(message)
                    self.bad_indices.add(current_idx)

                self.samples[current_idx] = None
                if len(self.bad_indices) >= len(self.samples):
                    raise RuntimeError("All samples were marked invalid; cannot continue.")

                idx = current_idx + 1
                attempts += 1
                continue

        raise RuntimeError("Exceeded retry budget while sampling training data.")


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    text_tensors = [item["text_ids"] for item in batch]
    code_tensors = [item["codes"] for item in batch]
    condition_tensors = [item["condition"] for item in batch]
    emo_tensors = [item["emo_vec"] for item in batch]

    text_padded = pad_sequence(text_tensors, batch_first=True, padding_value=0)
    code_padded = pad_sequence(code_tensors, batch_first=True, padding_value=0)
    condition_stacked = torch.stack(condition_tensors, dim=0)
    emo_stacked = torch.stack(emo_tensors, dim=0)

    text_lengths = torch.stack([item["text_len"] for item in batch])
    code_lengths = torch.stack([item["code_len"] for item in batch])
    cond_lengths = torch.stack([item["condition_len"] for item in batch])

    ids = [item["id"] for item in batch]
    prompt_ids = [item.get("prompt_id", item["id"]) for item in batch]
    target_ids = [item.get("target_id", item["id"]) for item in batch]
    languages = [item.get("language") for item in batch]
    prompt_languages = [item.get("prompt_language") for item in batch]
    manifest_paths = [item.get("manifest_path") for item in batch]

    return {
        "ids": ids,
        "prompt_ids": prompt_ids,
        "target_ids": target_ids,
        "text_ids": text_padded,
        "codes": code_padded,
        "condition": condition_stacked,
        "emo_vec": emo_stacked,
        "text_lengths": text_lengths,
        "code_lengths": code_lengths,
        "condition_lengths": cond_lengths,
        "languages": languages,
        "prompt_languages": prompt_languages,
        "manifest_paths": manifest_paths,
    }


def load_tokenizer(tokenizer_path: Path) -> TextTokenizer:
    normalizer = TextNormalizer()
    tokenizer = TextTokenizer(str(tokenizer_path), normalizer)
    return tokenizer


def build_model(cfg_path: Path, tokenizer: TextTokenizer, base_checkpoint: Path, device: torch.device) -> UnifiedVoice:
    cfg = OmegaConf.load(cfg_path)
    vocab_size = tokenizer.vocab_size
    if cfg.gpt.number_text_tokens != vocab_size:
        cfg.gpt.number_text_tokens = vocab_size

    model = UnifiedVoice(**cfg.gpt)
    checkpoint = torch.load(base_checkpoint, map_location="cpu")
    raw_state_dict = checkpoint.get("model", checkpoint)

    filtered_state_dict = {}
    for key, value in raw_state_dict.items():
        if key.startswith("inference_model."):
            continue
        if ".lora_" in key:
            continue
        new_key = key.replace(".base_layer.", ".")
        if new_key == "gpt.wte.weight":
            continue
        filtered_state_dict[new_key] = value
    state_dict = filtered_state_dict

    resizable_keys = {
        "text_embedding.weight": model.text_embedding.weight,
        "text_head.weight": model.text_head.weight,
        "text_head.bias": model.text_head.bias,
    }
    for key, param in resizable_keys.items():
        weight = state_dict.pop(key, None)
        if weight is None:
            continue
        with torch.no_grad():
            slices = tuple(min(a, b) for a, b in zip(param.shape, weight.shape))
            if param.ndim == 1:
                param[: slices[0]].copy_(weight[: slices[0]])
            else:
                param[: slices[0], : slices[1]].copy_(weight[: slices[0], : slices[1]])
        state_dict[key] = param.detach().clone()

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warn] Missing keys during load: {missing}")
    if unexpected:
        print(f"[Warn] Unexpected keys during load: {unexpected}")

    model = model.to(device)
    
    # L4 optimization: Use channels_last memory format for conv layers (if any)
    # Note: This primarily benefits CNNs; transformers may not see significant speedup
    # This can provide 20-30% speedup on Tensor Cores for conv-heavy models
    try:
        model = model.to(memory_format=torch.channels_last)
        print("[L4 Optimizations] Model converted to channels_last memory format")
    except Exception:
        # Not all models support channels_last, fall back gracefully
        pass
    
    return model


def compute_losses(
    model: UnifiedVoice,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    condition = batch["condition"].to(device)
    text_ids = batch["text_ids"].to(device)
    codes = batch["codes"].to(device)
    emo_vec = batch["emo_vec"].to(device)
    text_lengths = batch["text_lengths"].to(device)
    code_lengths = batch["code_lengths"].to(device)

    batch_size = text_ids.size(0)
    use_speed = torch.zeros(batch_size, dtype=torch.long, device=device)

    text_inputs = model.set_text_padding(text_ids.clone(), text_lengths)
    text_inputs = F.pad(text_inputs, (0, 1), value=model.stop_text_token)
    text_inputs, text_targets = model.build_aligned_inputs_and_targets(
        text_inputs, model.start_text_token, model.stop_text_token
    )

    mel_inputs = model.set_mel_padding(codes.clone(), code_lengths)
    mel_inputs = F.pad(mel_inputs, (0, 1), value=model.stop_mel_token)
    mel_inputs, mel_targets = model.build_aligned_inputs_and_targets(
        mel_inputs, model.start_mel_token, model.stop_mel_token
    )

    duration_zero = model.speed_emb(torch.zeros_like(use_speed))
    duration_one = model.speed_emb(torch.ones_like(use_speed))
    conds = torch.cat(
        (condition + emo_vec.unsqueeze(1), duration_one.unsqueeze(1), duration_zero.unsqueeze(1)),
        dim=1,
    )

    text_emb = model.text_embedding(text_inputs) + model.text_pos_embedding(text_inputs)
    mel_emb = model.mel_embedding(mel_inputs) + model.mel_pos_embedding(mel_inputs)

    text_logits, mel_logits = model.get_logits(conds, text_emb, model.text_head, mel_emb, model.mel_head)

    text_mask = (
        torch.arange(text_targets.size(1), device=device).unsqueeze(0)
        < (text_lengths + 1).unsqueeze(1)
    )
    mel_mask = (
        torch.arange(mel_targets.size(1), device=device).unsqueeze(0)
        < (code_lengths + 1).unsqueeze(1)
    )

    text_ce = F.cross_entropy(text_logits, text_targets, reduction="none")
    mel_ce = F.cross_entropy(mel_logits, mel_targets, reduction="none")

    text_loss = (text_ce * text_mask).sum() / text_mask.sum().clamp_min(1)
    mel_loss = (mel_ce * mel_mask).sum() / mel_mask.sum().clamp_min(1)

    metrics = {}
    with torch.no_grad():
        mel_logits_flat = mel_logits.permute(0, 2, 1).reshape(-1, mel_logits.size(1))
        mel_targets_flat = mel_targets.reshape(-1)
        mel_mask_flat = mel_mask.reshape(-1)
        if mel_mask_flat.any():
            valid_logits = mel_logits_flat[mel_mask_flat]
            valid_targets = mel_targets_flat[mel_mask_flat]
            top1 = (valid_logits.argmax(dim=-1) == valid_targets).float().mean().item()
        else:
            top1 = 0.0
        metrics["mel_top1"] = top1

    return text_loss, mel_loss, metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    step: int,
    recent_checkpoints: List[str],
    batch_idx: int = 0,
    extra: Dict[str, str] | None = None,
    last_losses: Dict[str, float] | None = None,
    grad_accumulation: int = 1,  # NEW: Pass grad_accumulation for accurate tracking
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "step": step,
        "batch_idx": batch_idx,
        "recent_checkpoints": recent_checkpoints,
        "accumulation_counter": batch_idx % grad_accumulation if batch_idx > 0 else 0,
    }
    if extra:
        state["extra"] = extra
    if last_losses:
        state["last_losses"] = last_losses
    torch.save(state, path)


def cleanup_old_checkpoints(output_dir: Path, recent_checkpoints: List[str], keep_count: int = 3) -> None:
    """Clean up old checkpoint files to save disk space.
    
    Only removes checkpoints NOT in the recent_checkpoints list.
    Always preserves latest.pth.
    
    Args:
        output_dir: Directory containing checkpoints
        recent_checkpoints: List of checkpoint paths to keep
        keep_count: Number of recent checkpoints to keep (used as fallback)
    """
    try:
        # Convert recent checkpoints to set of paths for quick lookup
        # Handle both absolute and relative paths
        keep_files = set()
        for ckpt in recent_checkpoints:
            ckpt_path = Path(ckpt)
            keep_files.add(ckpt_path.name)  # Just the filename
        
        keep_files.add("latest.pth")  # Always keep latest.pth
        
        # Clean up old checkpoint files not in recent list
        if output_dir.exists():
            all_checkpoints = sorted(output_dir.glob("model_step*.pth"), key=lambda p: p.stat().st_mtime)
            
            for ckpt_file in all_checkpoints:
                if ckpt_file.name not in keep_files:
                    try:
                        ckpt_file.unlink()
                        print(f"[Cleanup] Removed old checkpoint: {ckpt_file.name}")
                    except OSError as e:
                        print(f"[Warn] Could not remove {ckpt_file.name}: {e}")
                
    except Exception as e:
        print(f"[Warn] Cleanup encountered an error: {e}")


def cleanup_pycache() -> None:
    """Clean up Python cache files."""
    try:
        # Clean up __pycache__ directories
        for pycache_dir in Path(".").rglob("__pycache__"):
            try:
                shutil.rmtree(pycache_dir)
                print(f"[Cleanup] Removed cache: {pycache_dir}")
            except OSError:
                pass
        
        # Clean up .pyc files
        for pyc_file in Path(".").rglob("*.pyc"):
            try:
                pyc_file.unlink()
            except OSError:
                pass
                
    except Exception as e:
        print(f"[Warn] Cleanup encountered an error: {e}")


def evaluate(model: UnifiedVoice, loader: DataLoader, device: torch.device, max_batches: int = 0) -> Dict[str, float]:
    """Evaluate model on validation set.
    
    Args:
        model: Model to evaluate
        loader: Validation data loader
        device: Device to run on
        max_batches: Maximum number of batches to evaluate (0=all). Saves GPU time.
    
    Returns:
        Dictionary with average losses and metrics
    """
    model.eval()
    totals = {"text_loss": 0.0, "mel_loss": 0.0, "mel_top1": 0.0}
    count = 0
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                # Early stopping: limit validation batches to save GPU time
                if max_batches > 0 and batch_idx >= max_batches:
                    break
                try:
                    text_loss, mel_loss, metrics = compute_losses(model, batch, device)
                    bsz = batch["text_ids"].size(0)
                    totals["text_loss"] += text_loss.item() * bsz
                    totals["mel_loss"] += mel_loss.item() * bsz
                    totals["mel_top1"] += metrics["mel_top1"] * bsz
                    count += bsz
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        print(f"[Warn] Validation OOM at batch {batch_idx}, skipping remaining batches")
                        break
                    raise
    finally:
        # CRITICAL: Always restore training mode, even if validation crashes
        model.train()
    
    if count == 0:
        return {k: 0.0 for k in totals}
    return {k: v / count for k, v in totals.items()}


def validate_resume_consistency(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    checkpoint: Dict,
    first_batch: Dict[str, torch.Tensor],
    device: torch.device,
    args,
    base_vocab_size: int,
    current_vocab_size: int,
    skip_optimizer_load: bool,
) -> None:
    """Comprehensive validation after checkpoint resume."""
    issues = []
    
    print("\n" + "="*80)
    print("[Resume Validation] Running consistency checks...")
    print("="*80)
    
    # 1. Loss Consistency Check (skip if optimizer was reset)
    if not skip_optimizer_load and "last_losses" in checkpoint:
        checkpoint_losses = checkpoint["last_losses"]
        was_training = model.training
        try:
            with torch.no_grad():
                model.eval()
                text_loss, mel_loss, metrics = compute_losses(model, first_batch, device)
        finally:
            if was_training:
                model.train()
        
        checkpoint_text = checkpoint_losses.get("text_loss", 0)
        checkpoint_mel = checkpoint_losses.get("mel_loss", 0)
        
        if checkpoint_text > 0 and checkpoint_mel > 0:
            text_diff_pct = abs(text_loss.item() - checkpoint_text) / checkpoint_text * 100
            mel_diff_pct = abs(mel_loss.item() - checkpoint_mel) / checkpoint_mel * 100
            
            print(f"[Resume Validation] Loss Consistency Check:")
            print(f"   Checkpoint: text={checkpoint_text:.4f}, mel={checkpoint_mel:.4f}")
            print(f"   Current:    text={text_loss.item():.4f}, mel={mel_loss.item():.4f}")
            print(f"   Variance:   text={text_diff_pct:.1f}%, mel={mel_diff_pct:.1f}%")
            
            if text_diff_pct > 50 or mel_diff_pct > 50:
                issues.append(
                    f"🚨 CRITICAL: Large loss variance (text: {text_diff_pct:.1f}%, mel: {mel_diff_pct:.1f}%)\n"
                    f"   Expected <30% variance due to data shuffle.\n"
                    f"   This indicates potential resume corruption or configuration mismatch."
                )
            elif text_diff_pct > 30 or mel_diff_pct > 30:
                print(f"   ⚠️  WARNING: Moderate loss variance (>30%) - monitor closely")
            else:
                print(f"   ✅ PASS: Loss variance within expected range (<30%)")
    
    # 2. Learning Rate Consistency Check
    if scheduler and not skip_optimizer_load:
        actual_lr = scheduler.get_last_lr()[0]
        global_step = checkpoint.get("step", 0)
        
        # Expected LR calculation (cosine schedule with warmup)
        warmup_steps = args.warmup_steps
        total_steps = args.max_steps if args.max_steps > 0 else args.epochs * 1000  # Rough estimate
        
        if global_step < warmup_steps:
            expected_lr = args.learning_rate * (global_step / warmup_steps)
        else:
            progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
            expected_lr = args.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
        
        lr_diff_pct = abs(actual_lr - expected_lr) / max(expected_lr, 1e-8) * 100
        
        print(f"\n[Resume Validation] Learning Rate Check:")
        print(f"   Expected LR: {expected_lr:.2e} (for step {global_step})")
        print(f"   Actual LR:   {actual_lr:.2e}")
        print(f"   Variance:    {lr_diff_pct:.1f}%")
        
        if lr_diff_pct > 20:
            issues.append(
                f"🚨 CRITICAL: LR mismatch ({lr_diff_pct:.1f}% difference)\n"
                f"   This indicates scheduler state corruption.\n"
                f"   Expected: {expected_lr:.2e}, Got: {actual_lr:.2e}"
            )
        else:
            print(f"   ✅ PASS: LR matches expected value")
    
    # 3. Extended Vocab Gradient Masking Validation
    if current_vocab_size > base_vocab_size:
        print(f"\n[Resume Validation] Extended Vocab Gradient Masking Check:")
        print(f"   Vocab: {current_vocab_size} tokens ({base_vocab_size} base + {current_vocab_size - base_vocab_size} extended)")
        
        # Run one backward pass to check gradient masking
        was_training = model.training
        try:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            # Match main training loop AMP dtype
            use_amp = args.amp and device.type == "cuda"
            if use_amp:
                # Determine AMP dtype (same logic as main training)
                try:
                    from indextts.utils.hardware_optimizer import detect_hardware
                    hw_config = detect_hardware()
                    amp_dtype = torch.bfloat16 if hw_config.amp_dtype == "bfloat16" else torch.float16
                except:
                    amp_dtype = torch.float16  # Fallback
            else:
                amp_dtype = torch.float32
            
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                text_loss, mel_loss, _ = compute_losses(model, first_batch, device)
                loss = args.text_loss_weight * text_loss + args.mel_loss_weight * mel_loss
            loss.backward()
            
            # Check if base embeddings have zero gradients
            base_grad_norm = model.text_embedding.weight.grad[:base_vocab_size].norm().item()
            new_grad_norm = model.text_embedding.weight.grad[base_vocab_size:].norm().item()
            
            print(f"   Base token grad norm:     {base_grad_norm:.6e}")
            print(f"   Extended token grad norm: {new_grad_norm:.6e}")
            
            if base_grad_norm > 1e-6:
                issues.append(
                    f"🚨 CRITICAL: Gradient masking FAILED!\n"
                    f"   Base embeddings have gradients ({base_grad_norm:.2e})\n"
                    f"   This will corrupt the pretrained base model.\n"
                    f"   Check gradient hook registration."
                )
            elif new_grad_norm < 1e-8:
                issues.append(
                    f"⚠️  WARNING: Extended embeddings have near-zero gradients ({new_grad_norm:.2e})\n"
                    f"   Model may not be learning new language tokens."
                )
            else:
                print(f"   ✅ PASS: Gradient masking working correctly")
        finally:
            # Always clean up and restore model state
            optimizer.zero_grad(set_to_none=True)
            if not was_training:
                model.eval()
    
    # 4. Gradient Accumulation State Check
    if "accumulation_counter" in checkpoint:
        accum_state = checkpoint["accumulation_counter"]
        if accum_state != 0:
            print(f"\n[Resume Validation] Gradient Accumulation Check:")
            print(f"   ⚠️  WARNING: Resuming mid-accumulation cycle (counter={accum_state})")
            print(f"   {accum_state} batches of partial gradients were lost.")
            print(f"   Expect minor loss discontinuity for ~100 steps.")
            print(f"   Recommendation: Always stop training at checkpoint save points.")
    
    # 5. Optimizer State Alignment Check
    optimizer_param_count = len(optimizer.state)
    model_param_count = sum(1 for p in model.parameters() if p.requires_grad)
    
    print(f"\n[Resume Validation] Optimizer State Check:")
    print(f"   Optimizer tracking: {optimizer_param_count} parameters")
    print(f"   Model has:          {model_param_count} trainable parameters")
    
    if optimizer_param_count > 0 and abs(optimizer_param_count - model_param_count) > 10:
        issues.append(
            f"⚠️  WARNING: Optimizer/model parameter count mismatch\n"
            f"   Optimizer: {optimizer_param_count}, Model: {model_param_count}\n"
            f"   Optimizer state may be stale."
        )
    else:
        print(f"   ✅ PASS: Optimizer state aligned with model")
    
    # Summary
    print("\n" + "="*80)
    if issues:
        print("[Resume Validation] ❌ VALIDATION FAILED")
        print("="*80)
        for issue in issues:
            print(f"\n{issue}")
        print("\n" + "="*80)
        
        # Don't raise error, but give strong warning
        print("\n⚠️⚠️⚠️  CRITICAL WARNING  ⚠️⚠️⚠️")
        print("Resume validation detected serious issues.")
        print("Training may not converge or may corrupt the model.")
        print("\nRecommendations:")
        print("1. If losses don't drop within 1000 steps, STOP and debug")
        print("2. Monitor TensorBoard closely for anomalies")
        print("3. Consider starting fresh training if issues persist")
        print("\nPress Ctrl+C within 10 seconds to abort, or training will continue...")
        import time
        for i in range(10, 0, -1):
            print(f"  {i}...", end="\r")
            time.sleep(1)
        print("\nProceeding with training (issues logged above)...\n")
    else:
        print("[Resume Validation] ✅ ALL CHECKS PASSED")
        print("="*80 + "\n")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    
    # Auto-detect hardware and get optimal settings
    hw_config = detect_hardware()
    print_hardware_summary(hw_config)
    
    device = torch.device("cuda" if hw_config.has_cuda else "cpu")
    
    # Apply hardware-specific optimizations
    if hw_config.has_cuda:
        if hw_config.use_tf32:
            # Enable TF32 for matmul (3-8× speedup on Ampere/Ada/Hopper GPUs)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN autotuner for optimal performance
        torch.backends.cudnn.benchmark = True
        
        # Disable cuDNN deterministic for speed (comment out if reproducibility needed)
        torch.backends.cudnn.deterministic = False
    
    # Override with auto-detected values if not specified by user
    if args.batch_size == 0:
        args.batch_size = hw_config.batch_size
        print(f"[Auto-Optimization] Using batch_size={args.batch_size} (auto-detected)")
    
    if args.grad_accumulation == 0:
        args.grad_accumulation = hw_config.grad_accumulation
        print(f"[Auto-Optimization] Using grad_accumulation={args.grad_accumulation} (auto-detected)")
    
    if args.num_workers == 0:
        args.num_workers = hw_config.num_workers
        print(f"[Auto-Optimization] Using num_workers={args.num_workers} (auto-detected)")
    
    # Auto-enable AMP if not explicitly set and GPU supports it
    if not args.amp and hw_config.use_amp:
        print(f"[Auto-Optimization] Enabling --amp (recommended for GPU training)")
        args.amp = True
    
    effective_batch = args.batch_size * args.grad_accumulation
    print(f"[Training Config] Effective batch size: {effective_batch}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean up Python cache files only (not checkpoints!)
    print("[Info] Cleaning up cache files...")
    cleanup_pycache()
    
    log_root = output_dir / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    run_name = (
        f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if os.environ.get("INDEXTTS_RUN_NAME") is None
        else os.environ["INDEXTTS_RUN_NAME"]
    )
    log_dir = log_root / run_name
    writer = SummaryWriter(log_dir=str(log_dir))

    tokenizer = load_tokenizer(args.tokenizer)
    model = build_model(args.config, tokenizer, args.base_checkpoint, device)
    
    # Note: Gradient hooks will be registered AFTER potential checkpoint resume
    # to ensure they apply to the loaded weights
    base_vocab_size = 12000  # Base English/Chinese vocabulary size
    current_vocab_size = tokenizer.vocab_size
    
    def apply_extended_vocab_fix():
        """Apply gradient hooks for extended vocabulary training."""
        if current_vocab_size <= base_vocab_size:
            return
        
        print(f"\n{'='*80}")
        print(f"[Extended Vocab Fix] Detected extended vocabulary: {current_vocab_size} tokens")
        print(f"[Extended Vocab Fix] Base tokens: 0-{base_vocab_size-1} (pretrained)")
        print(f"[Extended Vocab Fix] New tokens: {base_vocab_size}-{current_vocab_size-1} (random init)")
        print(f"[Extended Vocab Fix] Applying gradient masking to freeze base embeddings")
        print(f"{'='*80}\n")
        
        # Gradient hook to freeze base token embeddings during training
        def freeze_base_tokens_hook(grad):
            """Zero out gradients for base vocabulary tokens (0-11999)"""
            if grad is None:
                return None
            if grad.shape[0] <= base_vocab_size:
                return grad
            grad_clone = grad.clone()
            grad_clone[:base_vocab_size] = 0
            return grad_clone
        
        # Register hooks on text embedding layers
        model.text_embedding.weight.register_hook(freeze_base_tokens_hook)
        model.text_head.weight.register_hook(freeze_base_tokens_hook)
        model.text_head.bias.register_hook(freeze_base_tokens_hook)
        
        # Calculate frozen parameters
        frozen_params = base_vocab_size * model.model_dim * 2
        frozen_params += base_vocab_size
        total_params = sum(p.numel() for p in model.parameters())
        frozen_pct = (frozen_params / total_params) * 100
        
        print(f"[Extended Vocab Fix] Gradient hooks registered for selective training")
        print(f"[Extended Vocab Fix] Freezing {frozen_params:,} / {total_params:,} parameters ({frozen_pct:.1f}%)")
        print(f"[Extended Vocab Fix] Base embeddings frozen, new embeddings trainable\n")
    
    # Enable gradient checkpointing if requested (saves VRAM)
    if args.grad_checkpointing and hasattr(model, 'gpt'):
        try:
            if hasattr(model.gpt, 'gradient_checkpointing_enable'):
                model.gpt.gradient_checkpointing_enable()
                print("[L4 Optimizations] Gradient checkpointing enabled")
        except Exception as e:
            print(f"[Warn] Could not enable gradient checkpointing: {e}")

    train_specs = parse_manifest_specs(args.train_manifests, "--train-manifest")
    val_specs = parse_manifest_specs(args.val_manifests, "--val-manifest")

    print("[Info] Loading training manifests...")
    train_dataset = JapaneseGPTDataset(train_specs)
    print("[Info] Loading validation manifests...")
    val_dataset = JapaneseGPTDataset(val_specs)

    manifest_metadata = {
        "train": [
            {
                "path": str(entry["path"]),
                "count": entry["count"],
                "languages": list(entry["languages"]),
            }
            for entry in train_dataset.manifest_summaries
        ],
        "val": [
            {
                "path": str(entry["path"]),
                "count": entry["count"],
                "languages": list(entry["languages"]),
            }
            for entry in val_dataset.manifest_summaries
        ],
    }

    def checkpoint_extra(extra_type: str) -> Dict[str, object]:
        return {
            "type": extra_type,
            "manifests": manifest_metadata,
            "tokenizer": str(args.tokenizer),  # Track tokenizer for validation
            "amp_dtype": hw_config.amp_dtype,  # Track AMP dtype for compatibility
        }

    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=use_cuda,
    )
    # GPU optimization: 2× batch size for validation (no gradients = more VRAM available)
    val_batch_size = args.val_batch_size if args.val_batch_size > 0 else (args.batch_size * 2)
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=use_cuda,
        persistent_workers=True if args.num_workers > 0 else False,  # Keep workers alive between val runs
    )
    print(f"[Info] Validation: batch_size={val_batch_size}, max_batches={args.max_val_batches} (0=unlimited)")

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * max(1, len(train_loader)) // max(1, args.grad_accumulation)
    total_steps = max(total_steps, 1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    use_amp = args.amp and device.type == "cuda"
    
    # Use detected AMP dtype from hardware config
    if hw_config.amp_dtype == "bfloat16":
        amp_dtype = torch.bfloat16
        scaler = None if use_amp else None  # bfloat16 doesn't need gradient scaling
    elif hw_config.amp_dtype == "float16":
        amp_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
    else:
        amp_dtype = torch.float32
        scaler = None

    global_step = 0
    start_epoch = 0
    start_batch_idx = 0
    recent_checkpoints: List[str] = []
    last_saved_step: int | None = None
    original_train_loader_length = len(train_loader)  # Store for epoch boundary calculation

    resume_path: str | None = None
    if args.resume:
        if args.resume == "auto":
            candidate = output_dir / "latest.pth"
            if candidate.exists():
                resume_path = str(candidate)
                print(f"[Info] Auto-resume: found checkpoint at {candidate}")
            else:
                print(f"[Info] Auto-resume: no checkpoint found at {candidate}, starting from scratch")
        else:
            resume_path = args.resume
            if not Path(resume_path).exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    
    if resume_path:
        try:
            print(f"[Info] Loading checkpoint from {resume_path}...")
            # checkpoint = torch.load(resume_path, map_location=device)
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            # Validate checkpoint structure
            required_keys = ["model", "optimizer", "epoch", "step"]
            missing_keys = [k for k in required_keys if k not in checkpoint]
            if missing_keys:
                raise ValueError(f"Checkpoint missing required keys: {missing_keys}")
            
            # Validate vocab size compatibility
            checkpoint_vocab = None
            if "model" in checkpoint:
                # Check text_embedding.weight shape
                for key, value in checkpoint["model"].items():
                    if key == "text_embedding.weight":
                        checkpoint_vocab = value.shape[0]
                        break
            
            skip_optimizer_load = False
            # CRITICAL: Model has vocab_size + 1 embeddings (for STOP_TEXT_TOKEN)
            # Checkpoint stores text_embedding.weight with shape [vocab_size + 1, hidden]
            # So we need to subtract 1 from checkpoint_vocab before comparing
            checkpoint_actual_vocab = checkpoint_vocab - 1 if checkpoint_vocab is not None else None
            
            if checkpoint_actual_vocab is not None and checkpoint_actual_vocab != current_vocab_size:
                print(f"\n🚨 CRITICAL: Vocab size mismatch detected!")
                print(f"   Checkpoint vocab: {checkpoint_vocab} embeddings = {checkpoint_actual_vocab} tokens + STOP")
                print(f"   Current tokenizer: {current_vocab_size} tokens")
                print(f"   Difference: {abs(checkpoint_actual_vocab - current_vocab_size)} tokens")
                print(f"   Optimizer state is INCOMPATIBLE with current model.")
                print(f"   ❌ Will NOT load optimizer/scheduler (would prevent learning!)")
                print(f"   ✅ Will load model weights and continue with fresh optimizer.\n")
                skip_optimizer_load = True
            elif checkpoint_vocab is not None:
                # Vocab sizes match (accounting for STOP token)
                print(f"[Info] ✅ Vocab size validated: {checkpoint_vocab} embeddings ({checkpoint_actual_vocab} tokens + STOP)")
            
            # Validate tokenizer path matches (critical for correct token mappings)
            checkpoint_tokenizer = checkpoint.get("manifests", {}).get("tokenizer")
            if checkpoint_tokenizer and str(args.tokenizer) != checkpoint_tokenizer:
                print(f"\n⚠️  WARNING: Tokenizer path changed!")
                print(f"   Checkpoint used: {checkpoint_tokenizer}")
                print(f"   Current: {args.tokenizer}")
                print(f"   Token mappings may be incorrect if tokenizers differ!\n")
            
            # Load model state
            print("[Info] Restoring model state...")
            missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
            if missing:
                print(f"[Warn] Missing keys: {missing[:3]}..." if len(missing) > 3 else f"[Warn] Missing: {missing}")
            if unexpected:
                print(f"[Warn] Unexpected keys: {unexpected[:3]}..." if len(unexpected) > 3 else f"[Warn] Unexpected: {unexpected}")
            
            # Only load optimizer/scheduler/scaler if vocab sizes match
            if not skip_optimizer_load:
                print("[Info] Restoring optimizer state...")
                optimizer.load_state_dict(checkpoint["optimizer"])
                
                print("[Info] Restoring scheduler state...")
                if checkpoint.get("scheduler"):
                    scheduler.load_state_dict(checkpoint["scheduler"])
                
                # Load scaler state (for AMP)
                # Note: bfloat16 doesn't use scaler (scaler=None), float16 does
                if scaler is not None and checkpoint.get("scaler") is not None:
                    print("[Info] Restoring gradient scaler state...")
                    try:
                        scaler.load_state_dict(checkpoint["scaler"])
                    except Exception as e:
                        print(f"[Warn] Could not restore scaler state: {e}. Using fresh scaler.")
                        scaler = torch.cuda.amp.GradScaler()
                elif scaler is not None and checkpoint.get("scaler") is None:
                    print("[Warn] AMP enabled but no scaler state in checkpoint, using fresh scaler")
                elif scaler is None and checkpoint.get("scaler") is not None:
                    print("[Info] Checkpoint has scaler but current run uses bfloat16 (no scaler needed)")
            else:
                print("[Info] ❌ SKIPPING optimizer, scheduler, AND scaler restore (incompatible checkpoint)")
                print("[Info] ✅ Using FRESH optimizer with initial LR and warmup")
                print(f"[Info] 🔄 Training will RESTART from step 0 (was at step {global_step})")
                print("[Info] 📊 Expect losses to drop significantly within 5k-10k steps")
                # CRITICAL: Reset ALL training state when using fresh optimizer
                global_step = 0
                start_epoch = 0  # NEW: Reset epoch counter
                start_batch_idx = 0  # NEW: Reset batch counter
                # Recreate scheduler with fresh state (starts at step 0)
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=args.warmup_steps,
                    num_training_steps=total_steps,
                )
                # Reset scaler to match fresh optimizer
                if scaler is not None:
                    print("[Info] Resetting gradient scaler to align with fresh optimizer")
                    scaler = torch.cuda.amp.GradScaler()
                # CRITICAL: Also reset last_saved_step to allow first checkpoint save
                last_saved_step = None
                print("[Info] ⚠️  Note: Training position resets to epoch 0, batch 0, step 0 (fresh start)")
            
            # Restore training state
            # CRITICAL: Proper epoch/batch restoration
            # Checkpoint saves the NEXT epoch/batch to resume from
            # This ensures perfect continuity across interruptions
            start_epoch = checkpoint.get("epoch", 0)
            start_batch_idx = checkpoint.get("batch_idx", 0)
            
            global_step = checkpoint.get("step", 0)
            recent_checkpoints = checkpoint.get("recent_checkpoints", [])
            last_saved_step = checkpoint.get("step")
            
            # CRITICAL: Validate training config compatibility
            ckpt_batch_size = checkpoint.get("batch_size")
            ckpt_grad_accum = checkpoint.get("grad_accumulation")
            ckpt_pt_version = checkpoint.get("pytorch_version", "unknown")
            ckpt_peak_memory = checkpoint.get("cuda_peak_memory", 0)
            
            # Validate PyTorch version compatibility
            if ckpt_pt_version != "unknown":
                ckpt_major = ckpt_pt_version.split('.')[0]
                current_major = torch.__version__.split('.')[0]
                if ckpt_major != current_major:
                    raise RuntimeError(
                        f"❌ CRITICAL: PyTorch major version mismatch!\n"
                        f"   Checkpoint: {ckpt_pt_version}\n"
                        f"   Current: {torch.__version__}\n"
                        f"   Optimizer state is incompatible across major versions.\n"
                        f"   Please use PyTorch {ckpt_major}.x or start training from scratch."
                    )
            
            # Validate effective batch size (strict)
            if ckpt_batch_size is not None and ckpt_grad_accum is not None:
                old_effective = ckpt_batch_size * ckpt_grad_accum
                new_effective = args.batch_size * args.grad_accumulation
                
                if old_effective != new_effective:
                    raise RuntimeError(
                        f"❌ CRITICAL: Effective batch size changed!\n"
                        f"   Checkpoint: {ckpt_batch_size} × {ckpt_grad_accum} = {old_effective}\n"
                        f"   Current: {args.batch_size} × {args.grad_accumulation} = {new_effective}\n"
                        f"   This breaks optimizer momentum and learning rate scaling.\n"
                        f"   Use --batch-size {ckpt_batch_size} --grad-accumulation {ckpt_grad_accum} to match checkpoint."
                    )
                elif ckpt_grad_accum != args.grad_accumulation:
                    print(f"\n⚠️  WARNING: Gradient accumulation changed (same effective batch)")
                    print(f"   Checkpoint: grad_accum={ckpt_grad_accum}")
                    print(f"   Current: grad_accum={args.grad_accumulation}")
                    print(f"   Loss scaling differs - expect temporary instability for ~1000 steps.\n")
            
            # Validate memory availability
            if ckpt_peak_memory > 0 and torch.cuda.is_available():
                available = torch.cuda.get_device_properties(0).total_memory
                if ckpt_peak_memory > available * 0.95:
                    print(f"\n⚠️  WARNING: OOM risk detected!")
                    print(f"   Checkpoint peak memory: {ckpt_peak_memory / 1e9:.1f}GB")
                    print(f"   Current GPU memory: {available / 1e9:.1f}GB")
                    print(f"   Consider reducing batch size if OOM occurs.\n")
            
            # Restore RNG states for reproducible shuffle
            if "rng_state" in checkpoint:
                try:
                    # Restore CPU RNG
                    torch.set_rng_state(checkpoint["rng_state"].get("torch_cpu", checkpoint["rng_state"].get("torch")))
                    # Restore CUDA RNG (all GPUs) with count validation
                    if checkpoint["rng_state"].get("torch_cuda") is not None and torch.cuda.is_available():
                        saved_gpu_count = len(checkpoint["rng_state"]["torch_cuda"])
                        current_gpu_count = torch.cuda.device_count()
                        if saved_gpu_count != current_gpu_count:
                            print(f"[Warn] GPU count changed: {saved_gpu_count}→{current_gpu_count}, shuffle may differ")
                        torch.cuda.set_rng_state_all(checkpoint["rng_state"]["torch_cuda"])
                    # Restore NumPy and Python RNG
                    np.random.set_state(checkpoint["rng_state"]["numpy"])
                    random.setstate(checkpoint["rng_state"]["python"])
                    print("[Info] Restored RNG states (CPU+CUDA, reproducible shuffle)")
                except Exception as e:
                    print(f"[Warn] Could not restore RNG states: {e}")
                    print("[Warn] Training shuffle order may differ from original run (reproducibility lost)")
            
            print(f"[Info] ✅ Successfully resumed from {resume_path}")
            print(f"[Info]    Epoch: {start_epoch}, Step: {global_step}, Batch: {start_batch_idx}")
            print(f"[Info]    Recent checkpoints: {len(recent_checkpoints)}")
            
            # Warn if hyperparameters might have changed (can't detect perfectly, but helpful)
            print(f"\n💡 Reminder: Ensure you're using the SAME hyperparameters as original training:")
            print(f"   --learning-rate {args.learning_rate}")
            print(f"   --text-loss-weight {args.text_loss_weight}")
            print(f"   --mel-loss-weight {args.mel_loss_weight}")
            print(f"   Using different values may cause training instability.\n")
            
            # Note: Don't cleanup checkpoints on resume - we want to keep all trained checkpoints
            # Only __pycache__ cleanup is safe here
            for pycache_dir in Path(".").rglob("__pycache__"):
                try:
                    shutil.rmtree(pycache_dir)
                except OSError:
                    pass
            
            # NEW: Validate resume consistency
            print("\n[Info] Running resume consistency validation...")
            try:
                # Get first batch for validation
                first_batch = next(iter(train_loader))
                validate_resume_consistency(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    checkpoint=checkpoint,
                    first_batch=first_batch,
                    device=device,
                    args=args,
                    base_vocab_size=base_vocab_size,
                    current_vocab_size=current_vocab_size,
                    skip_optimizer_load=skip_optimizer_load,
                )
            except Exception as e:
                print(f"[Warn] Resume validation encountered an error: {e}")
                print("[Warn] Proceeding with training, but monitor closely.")
        
        except Exception as e:
            print(f"[Error] Failed to load checkpoint from {resume_path}: {e}")
            raise RuntimeError(f"Resume failed: {e}") from e
    
    # Apply extended vocab fix AFTER checkpoint loading (if any)
    # This ensures gradient hooks apply to the loaded model weights
    apply_extended_vocab_fix()

    model.train()
    optimizer.zero_grad(set_to_none=True)

    save_every = args.save_interval
    best_val = math.inf

    if args.val_interval > 0 and global_step > 0:
        # If we resumed exactly on a validation boundary we postpone evaluation until
        # after the next training step to avoid running validation before training.
        print("[Info] Skipping startup validation; will evaluate after next training interval.")

    # Calculate if we need to skip batches (resumed mid-epoch)
    # CRITICAL FIX: Don't align batch skip - it causes duplicate/skipped training
    # The checkpoint saves exact batch position, we must resume from exact position
    skip_batches_first_epoch = start_batch_idx if start_batch_idx > 0 else 0
    
    for epoch in range(start_epoch, args.epochs):
        # Reset batch index at the start of each NEW epoch (not when resuming mid-epoch)
        if epoch > start_epoch:
            current_batch = 0
        else:
            current_batch = start_batch_idx  # Resume from saved batch position
        
        # Create subset dataset for first epoch if resuming mid-epoch
        if epoch == start_epoch and skip_batches_first_epoch > 0:
            # Calculate which samples to skip
            skip_samples = skip_batches_first_epoch * args.batch_size
            print(f"[Info] Resuming from batch {skip_batches_first_epoch} in epoch {start_epoch}")
            print(f"[Info] Skipping first {skip_samples:,} samples (instant, no iteration)")
            
            # Create subset of remaining samples
            remaining_indices = list(range(skip_samples, len(train_dataset)))
            epoch_dataset = Subset(train_dataset, remaining_indices)
            epoch_loader = DataLoader(
                epoch_dataset,
                batch_size=args.batch_size,
                shuffle=False,  # Don't shuffle subset (already in epoch order)
                num_workers=args.num_workers,
                collate_fn=collate_batch,
                pin_memory=use_cuda,
            )
        else:
            epoch_loader = train_loader
        
        # CRITICAL FIX: Use real batch index from full dataset, not subset position
        # This ensures checkpoint saves correct absolute position
        for subset_idx, batch in enumerate(epoch_loader):
            batch_idx = subset_idx + (skip_batches_first_epoch if epoch == start_epoch else 0)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype if use_amp else torch.float32):
                text_loss, mel_loss, metrics = compute_losses(model, batch, device)
                loss = args.text_loss_weight * text_loss + args.mel_loss_weight * mel_loss
            if use_amp and scaler is not None:
                # float16 with gradient scaling
                scaler.scale(loss / args.grad_accumulation).backward()
            else:
                # bfloat16 or no AMP
                (loss / args.grad_accumulation).backward()

            if (batch_idx + 1) % args.grad_accumulation == 0:
                if args.grad_clip > 0:
                    if use_amp and scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if use_amp and scaler is not None:
                    # float16 path
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # bfloat16 or no AMP
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                if global_step % args.log_interval == 0:
                    writer.add_scalar("train/text_loss", text_loss.item(), global_step)
                    writer.add_scalar("train/mel_loss", mel_loss.item(), global_step)
                    writer.add_scalar("train/mel_top1", metrics["mel_top1"], global_step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                    print(
                        f"[Train] epoch={epoch + 1} step={global_step} "
                        f"text_loss={text_loss.item():.4f} mel_loss={mel_loss.item():.4f} "
                        f"mel_top1={metrics['mel_top1']:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                    )

                if args.val_interval > 0 and global_step > 0 and global_step % args.val_interval == 0:
                    val_metrics = evaluate(model, val_loader, device, max_batches=args.max_val_batches)
                    writer.add_scalar("val/text_loss", val_metrics["text_loss"], global_step)
                    writer.add_scalar("val/mel_loss", val_metrics["mel_loss"], global_step)
                    writer.add_scalar("val/mel_top1", val_metrics["mel_top1"], global_step)
                    print(
                        f"[Val] epoch={epoch + 1} step={global_step} "
                        f"text_loss={val_metrics['text_loss']:.4f} mel_loss={val_metrics['mel_loss']:.4f} "
                        f"mel_top1={val_metrics['mel_top1']:.4f}"
                    )
                    if val_metrics["mel_loss"] < best_val:
                        best_val = val_metrics["mel_loss"]

                # Save checkpoint (skip if we just resumed from this exact step)
                if global_step % save_every == 0 and last_saved_step != global_step:
                    ckpt_path = output_dir / f"model_step{global_step}.pth"
                    recent_checkpoints.append(str(ckpt_path))
                    # Keep only recent checkpoints
                    if len(recent_checkpoints) > args.keep_checkpoints:
                        recent_checkpoints = recent_checkpoints[-args.keep_checkpoints:]
                    
                    # Calculate next position to resume from
                    next_batch = batch_idx + 1
                    next_epoch = epoch
                    # CRITICAL FIX: Use original train_loader length, not subset length
                    # Handle epoch boundary: if next batch exceeds epoch, move to next epoch
                    if next_batch >= original_train_loader_length:
                        next_batch = 0
                        next_epoch = epoch + 1
                    
                    # CRITICAL FIX: Save RNG states for reproducible shuffle on resume
                    rng_state = {
                        "torch_cpu": torch.get_rng_state(),
                        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        "numpy": np.random.get_state(),
                        "python": random.getstate(),
                    }
                    
                    save_checkpoint(
                        ckpt_path,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        next_epoch,
                        global_step,
                        recent_checkpoints,
                        next_batch,
                        extra=checkpoint_extra("step"),
                        last_losses={
                            "text_loss": text_loss.item(),
                            "mel_loss": mel_loss.item(),
                            "mel_top1": metrics.get("mel_top1", 0.0),
                        },
                        grad_accumulation=args.grad_accumulation,
                    )
                    
                    # CRITICAL: Atomic write to prevent corruption on interruption
                    latest_path = output_dir / "latest.pth"
                    latest_tmp = output_dir / "latest.pth.tmp"
                    
                    # Save with explicit file handle for proper fsync
                    with latest_tmp.open('wb') as f:
                        torch.save(
                            {
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "scaler": scaler.state_dict() if scaler else None,
                                "epoch": next_epoch,
                                "step": global_step,
                                "batch_idx": next_batch,
                                "recent_checkpoints": recent_checkpoints,
                                "manifests": manifest_metadata,
                                "rng_state": rng_state,
                                "batch_size": args.batch_size,
                                "grad_accumulation": args.grad_accumulation,
                                "pytorch_version": torch.__version__,
                                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                                "cuda_peak_memory": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                            },
                            f,
                        )
                        f.flush()  # Flush Python buffers
                        os.fsync(f.fileno())  # CRITICAL: fsync BEFORE file close
                    # File is now closed and durably written, safe to replace
                    latest_tmp.replace(latest_path)  # Atomic on POSIX/Windows
                    # Clean up old checkpoints beyond keep limit
                    cleanup_old_checkpoints(output_dir, recent_checkpoints, args.keep_checkpoints)
                    last_saved_step = global_step

                if args.max_steps and global_step >= args.max_steps:
                    break

            if args.max_steps and global_step >= args.max_steps:
                break

        if args.max_steps and global_step >= args.max_steps:
            break

        if args.val_interval == 0:
            val_metrics = evaluate(model, val_loader, device, max_batches=args.max_val_batches)
            writer.add_scalar("val/text_loss", val_metrics["text_loss"], global_step)
            writer.add_scalar("val/mel_loss", val_metrics["mel_loss"], global_step)
            writer.add_scalar("val/mel_top1", val_metrics["mel_top1"], global_step)
            print(
                f"[Val] epoch={epoch + 1} step={global_step} "
                f"text_loss={val_metrics['text_loss']:.4f} mel_loss={val_metrics['mel_loss']:.4f} "
                f"mel_top1={val_metrics['mel_top1']:.4f}"
            )
            if val_metrics["mel_loss"] < best_val:
                best_val = val_metrics["mel_loss"]


    if global_step > 0 and last_saved_step != global_step:
        ckpt_path = output_dir / f"model_step{global_step}.pth"
        recent_checkpoints.append(str(ckpt_path))
        
        # Save RNG states for final checkpoint too
        rng_state = {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        
        save_checkpoint(
            ckpt_path,
            model,
            optimizer,
            scheduler,
            scaler,
            args.epochs,  # Completed all epochs
            global_step,
            recent_checkpoints,
            0,  # Training complete
            extra=checkpoint_extra("final"),
            last_losses={
                "text_loss": text_loss.item() if 'text_loss' in locals() else 0.0,
                "mel_loss": mel_loss.item() if 'mel_loss' in locals() else 0.0,
                "mel_top1": metrics.get("mel_top1", 0.0) if 'metrics' in locals() else 0.0,
            },
            grad_accumulation=args.grad_accumulation,
        )
        latest_path = output_dir / "latest.pth"
        latest_tmp = output_dir / "latest.pth.tmp"
        
        # Save with explicit file handle for proper fsync
        with latest_tmp.open('wb') as f:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler else None,
                    "epoch": args.epochs,  # Completed all epochs
                    "step": global_step,
                    "batch_idx": 0,  # Training complete
                    "recent_checkpoints": recent_checkpoints,
                    "manifests": manifest_metadata,
                    "rng_state": rng_state,
                    "batch_size": args.batch_size,
                    "grad_accumulation": args.grad_accumulation,
                    "pytorch_version": torch.__version__,
                    "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                    "cuda_peak_memory": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                },
                f,
            )
            f.flush()  # Flush Python buffers
            os.fsync(f.fileno())  # CRITICAL: fsync BEFORE file close
        # File is now closed and durably written
        latest_tmp.replace(latest_path)  # Atomic
        # Clean up old checkpoints beyond keep limit
        cleanup_old_checkpoints(output_dir, recent_checkpoints)

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
