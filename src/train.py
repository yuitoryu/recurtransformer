"""Train tokenized JSONL: python -B -m src.train --config config/train.json.

Relative paths resolve from the project root. Set checkpoint.resume_from to a
trusted .pt checkpoint to continue its original training schedule. Logs use
optimizer steps and token-weighted losses. Validation uses the entire prepared
validation JSONL. An interval of zero disables that
periodic action; final training metrics are still written. keep_last_n applies
only to periodic checkpoints, so zero retains none of them.
torch.compile's default backend additionally requires a C/C++ toolchain.
"""

import argparse
from array import array
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import threading
import time
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def log(message):
    print(f"{message} [{datetime.now():%Y-%m-%d %H:%M:%S}]", flush=True)


def configure_environment():
    """Keep library caches, compiler products and temporary files in the project."""
    cache = PROJECT_ROOT / ".cache"
    locations = {
        "TMPDIR": cache / "tmp", "TMP": cache / "tmp", "TEMP": cache / "tmp",
        "XDG_CACHE_HOME": cache,
        "MPLCONFIGDIR": cache / "matplotlib",
        "TORCH_HOME": cache / "torch",
        "TORCHINDUCTOR_CACHE_DIR": cache / "torchinductor",
        "TORCH_EXTENSIONS_DIR": cache / "torch_extensions",
        "TRITON_CACHE_DIR": cache / "triton",
        "CUDA_CACHE_PATH": cache / "cuda",
    }
    for key, path in locations.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    tempfile.tempdir = str(locations["TMPDIR"])


def project_path(value):
    return (PROJECT_ROOT / Path(value)).resolve()


class TerminalLog:
    """Tee OS stdout/stderr, including native/compiler output, to a file."""

    def __init__(self, path, to_console):
        self.path = path
        self.to_console = to_console

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self.output = self.path.open("ab", buffering=0)
        self.original_stdout = os.dup(1)
        self.original_stderr = os.dup(2)
        self.reader, writer = os.pipe()
        os.dup2(writer, 1)
        os.dup2(writer, 2)
        os.close(writer)
        self.thread = threading.Thread(target=self._copy, daemon=True)
        self.thread.start()
        return self

    def _copy(self):
        while chunk := os.read(self.reader, 65536):
            self.output.write(chunk)
            if self.to_console:
                remaining = memoryview(chunk)
                while remaining:
                    remaining = remaining[os.write(self.original_stdout, remaining):]

    def __exit__(self, *_):
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self.original_stdout, 1)
        os.dup2(self.original_stderr, 2)
        self.thread.join()
        os.close(self.reader)
        os.close(self.original_stdout)
        os.close(self.original_stderr)
        self.output.close()


class JsonlDataset:
    """Store row offsets and lengths; decode token lists only when requested."""

    def __init__(self, path, max_seq_len):
        self.path = path
        self.offsets = array("Q")
        self.lengths = array("I")
        skipped = 0
        offset = 0
        log(f"Indexing {path}")
        with path.open("rb") as source:
            for line_number, line in enumerate(source, 1):
                length = len(json.loads(line))
                if length > max_seq_len:
                    raise ValueError(f"{path}:{line_number}: sequence length {length} exceeds "
                                     f"model max_seq_len={max_seq_len}")
                if length >= 2:
                    self.offsets.append(offset)
                    self.lengths.append(length)
                else:
                    skipped += 1
                offset += len(line)
        if not self.offsets:
            raise ValueError(f"No sequences with at least two tokens in {path}")
        log(f"Indexed {len(self):,} sequences; skipped {skipped:,} short rows")
        self.source = path.open("rb")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        self.source.seek(self.offsets[index])
        return json.loads(self.source.readline())

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.source.close()


def collate_sequences(sequences, pin_memory=False):
    import torch

    # Right padding cannot affect earlier valid positions under the causal mask.
    width = max(map(len, sequences)) - 1
    inputs = torch.zeros((len(sequences), width), dtype=torch.long, pin_memory=pin_memory)
    targets = torch.full(inputs.shape, -100, dtype=torch.long, pin_memory=pin_memory)
    for row, sequence in enumerate(sequences):
        length = len(sequence) - 1
        inputs[row, :length] = torch.tensor(sequence[:-1], dtype=torch.long)
        targets[row, :length] = torch.tensor(sequence[1:], dtype=torch.long)
    return inputs, targets


def prepare_window(dataset, indices, batch_size, pin_memory=False):
    """Group only within one update; use a private file handle for prefetch."""
    indices = sorted(indices, key=lambda i: dataset.lengths[i], reverse=True)
    batches = []
    with dataset.path.open("rb") as source:
        for start in range(0, len(indices), batch_size):
            sequences = []
            for index in indices[start:start + batch_size]:
                source.seek(dataset.offsets[index])
                sequences.append(json.loads(source.readline()))
            batches.append(collate_sequences(sequences, pin_memory=pin_memory))
    return batches


def epoch_order(size, seed, epoch):
    import torch

    return torch.randperm(size, generator=torch.Generator().manual_seed(seed + epoch))


def learning_rate_at(step, total_steps, training):
    peak = training["learning_rate"]
    warmup = training["warmup_steps"]
    if step <= warmup:
        return peak * step / warmup
    progress = min(1.0, max(0.0, (step - warmup - 1) / max(1, total_steps - warmup - 1)))
    minimum = training["min_learning_rate"]
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * progress))


def autocast_context(device, precision):
    import torch

    if precision == "fp32":
        return nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    return torch.autocast(device_type=device.type, dtype=dtype)


def summed_loss(model, inputs, targets, device, precision):
    import torch.nn.functional as F

    # Enqueue both copies before forward; pinned training batches avoid host waits.
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    with autocast_context(device, precision):
        logits = model(inputs)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1), ignore_index=-100, reduction="sum")


def evaluate(model, dataset, batch_size, device, precision):
    import torch

    was_training = model.training
    model.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tokens = 0
    try:
        with torch.inference_mode():
            for start in range(0, len(dataset), batch_size):
                sequences = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
                inputs, targets = collate_sequences(sequences, pin_memory=device.type == "cuda")
                loss_sum += summed_loss(model, inputs, targets, device, precision).double()
                tokens += sum(len(sequence) - 1 for sequence in sequences)
    finally:
        model.train(was_training)
    return loss_sum.item() / tokens, tokens


def capture_rng():
    import numpy as np
    import torch

    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_save(payload, destination):
    import torch

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def write_json(path, value):
    with path.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, ensure_ascii=False)
        output.write("\n")


def append_metric(path, record):
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, allow_nan=False) + "\n")


def rewrite_metrics(path, records):
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, allow_nan=False) + "\n")


def plot_losses(train_history, eval_history, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 6))
    for records, label in ((train_history, "Train loss"), (eval_history, "Eval loss")):
        if records:
            axis.plot([r["step"] for r in records], [r["loss"] for r in records], label=label, marker=".")
    axis.set(xlabel="Optimizer step", ylabel="Cross-entropy loss", title="Training and validation loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def validate_config(config):
    training = config["training"]
    if training["optimizer"].lower() != "adamw":
        raise ValueError("Only optimizer='adamw' is supported")
    if training["lr_scheduler"].lower() != "cosine":
        raise ValueError("Only lr_scheduler='cosine' is supported")
    if training["precision"] not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be fp32, bf16 or fp16")
    for key in ("num_epochs", "micro_batch_size", "gradient_accumulation_steps"):
        if training[key] <= 0:
            raise ValueError(f"training.{key} must be positive")
    if training["warmup_steps"] < 0:
        raise ValueError("warmup_steps cannot be negative")
    if not 0 <= training["min_learning_rate"] <= training["learning_rate"] or training["learning_rate"] <= 0:
        raise ValueError("Require 0 <= min_learning_rate <= learning_rate and learning_rate > 0")
    evaluation = config["evaluation"]
    if evaluation["eval_downstream"]:
        raise ValueError("Downstream evaluation is not implemented; use eval_downstream=false")
    if evaluation["eval_batch_size"] <= 0:
        raise ValueError("eval_batch_size must be positive")
    for section, key in (("logging", "log_interval_steps"), ("evaluation", "eval_interval_steps"),
                         ("checkpoint", "save_interval_steps"), ("checkpoint", "keep_last_n")):
        if config[section][key] < 0:
            raise ValueError(f"{section}.{key} cannot be negative")


def run_training(config, run_dir, log_dir):
    import numpy as np
    import torch
    from .model import RecurrentTransformer

    validate_config(config)
    experiment = config["experiment"]
    training = config["training"]
    evaluation = config["evaluation"]
    logging = config["logging"]
    checkpoint_config = config["checkpoint"]
    device = torch.device(config["system"]["device"])
    precision = training["precision"]
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable in this environment")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("The requested CUDA device does not support bf16")

    seed = experiment["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    with project_path(experiment["model_config_path"]).open(encoding="utf-8") as source:
        model_config = json.load(source)
    model_args = {key: value for key, value in model_config.items()
                  if key not in {"name", "use_weighted_from_each_loop", "use_regularizer"}}
    # model.py's device argument covers only its layers; move the complete model afterward.
    model_args["device"] = "cpu"
    model = RecurrentTransformer(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                 betas=(training["beta1"], training["beta2"]),
                                 eps=training["adam_eps"], weight_decay=training["weight_decay"])
    scaler = torch.amp.GradScaler(device.type, enabled=precision == "fp16")
    checkpoints = run_dir / "checkpoints"
    output_evaluation = run_dir / "evaluation"
    checkpoints.mkdir(exist_ok=True)
    output_evaluation.mkdir(exist_ok=True)
    write_json(log_dir / "model_config.json", model_config)

    with JsonlDataset(project_path(config["data"]["train_data_path"]), model.max_seq_len) as train_data, \
            JsonlDataset(project_path(config["data"]["val_data_path"]), model.max_seq_len) as val_data, \
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="train-prefetch") as prefetch:
        batch_size = training["micro_batch_size"]
        update_size = batch_size * training["gradient_accumulation_steps"]
        total_steps = math.ceil(len(train_data) / update_size) * training["num_epochs"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda index: learning_rate_at(index + 1, total_steps, training) / training["learning_rate"])
        val_tokens = sum(val_data.lengths) - len(val_data)
        state = {"epoch": 0, "next_sample": 0, "step": 0, "best_eval_loss": math.inf,
                 "train_history": [], "eval_history": [], "log_loss_sum": 0.0,
                 "log_tokens": 0, "log_samples": 0}
        resume_from = checkpoint_config["resume_from"]
        if resume_from:
            # Python/NumPy RNG states require full unpickling: use trusted training checkpoints.
            saved = torch.load(project_path(resume_from), map_location="cpu", weights_only=False)
            if (saved["config"]["training"] != training or saved["total_steps"] != total_steps
                    or saved["train_samples"] != len(train_data)
                    or saved["config"]["experiment"]["seed"] != seed):
                raise ValueError("Full resume requires the original training settings, seed and dataset size")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            scaler.load_state_dict(saved["scaler"])
            state = saved["state"]
            restore_rng(saved["rng"])
            log(f"Resumed {resume_from}: step={state['step']}, epoch={state['epoch']}, "
                f"next_sample={state['next_sample']}")
            write_json(log_dir / f"resume_config_step_{state['step']:08d}.json", config)
        else:
            write_json(log_dir / "train_config.json", config)

        train_metrics = log_dir / "train_metrics.jsonl"
        eval_metrics = log_dir / "eval_metrics.jsonl"
        rewrite_metrics(train_metrics, state["train_history"])
        rewrite_metrics(eval_metrics, state["eval_history"])
        trained_model = torch.compile(model, dynamic=True) if config["system"]["compile_model"] else model
        log(f"Device={device}, precision={precision}, parameters={sum(p.numel() for p in model.parameters()):,}, "
            f"updates={total_steps:,}")
        log(f"Validation: full dataset, {len(val_data):,} sequences, "
            f"{val_tokens:,} prediction tokens")

        def save_checkpoint(filename):
            payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                       "state": state, "rng": capture_rng(), "config": config,
                       "model_config": model_config, "total_steps": total_steps,
                       "train_samples": len(train_data)}
            atomic_save(payload, checkpoints / filename)
            log(f"Checkpoint: {checkpoints / filename}")

        def log_training():
            if not state["log_tokens"]:
                return
            record = {"step": state["step"], "epoch": state["epoch"] + state["next_sample"] / len(train_data),
                      "loss": state["log_loss_sum"] / state["log_tokens"], "learning_rate": last_lr,
                      "tokens": state["log_tokens"], "samples": state["log_samples"]}
            if logging["log_gpu_memory"] and device.type == "cuda":
                record["gpu_allocated_bytes"] = torch.cuda.memory_allocated(device)
                record["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            state["train_history"].append(record)
            append_metric(train_metrics, record)
            log(f"Train step={record['step']}/{total_steps} epoch={record['epoch']:.4f} "
                f"loss={record['loss']:.6f} lr={record['learning_rate']:.6g} tokens={record['tokens']:,}")
            state["log_loss_sum"] = 0.0
            state["log_tokens"] = 0
            state["log_samples"] = 0

        def run_evaluation():
            if state["eval_history"] and state["eval_history"][-1]["step"] == state["step"]:
                return
            loss, tokens = evaluate(trained_model, val_data, evaluation["eval_batch_size"], device, precision)
            record = {"step": state["step"], "loss": loss, "tokens": tokens}
            state["eval_history"].append(record)
            append_metric(eval_metrics, record)
            log(f"Eval step={state['step']} loss={loss:.6f} tokens={tokens:,}")
            if loss < state["best_eval_loss"]:
                state["best_eval_loss"] = loss
                if checkpoint_config["save_best"]:
                    save_checkpoint("best.pt")

        if evaluation["eval_at_start"] and state["step"] == 0:
            run_evaluation()
        started = time.monotonic()
        last_lr = optimizer.param_groups[0]["lr"]
        while state["epoch"] < training["num_epochs"]:
            epoch = state["epoch"]
            order = epoch_order(len(train_data), seed, epoch)
            model.train()
            start = state["next_sample"]
            pending = prefetch.submit(prepare_window, train_data,
                                      order[start:start + update_size].tolist(), batch_size,
                                      device.type == "cuda")
            while state["next_sample"] < len(train_data) and state["epoch"] == epoch:
                start = state["next_sample"]
                end = min(start + update_size, len(train_data))
                indices = order[start:end].tolist()
                batches = pending.result()
                # Bound host memory to the current and next update. Prefetch never
                # advances checkpoint progress or consumes training RNG state.
                if end < len(train_data):
                    pending = prefetch.submit(prepare_window, train_data,
                                              order[end:end + update_size].tolist(), batch_size,
                                              device.type == "cuda")
                token_count = sum(train_data.lengths[i] - 1 for i in indices)
                retry_rng = capture_rng() if scaler.is_enabled() else None
                while True:
                    optimizer.zero_grad(set_to_none=True)
                    window_loss = torch.zeros((), device=device, dtype=torch.float64)
                    for inputs, targets in batches:
                        loss = summed_loss(trained_model, inputs, targets, device, precision)
                        window_loss += loss.detach().double()
                        scaler.scale(loss / token_count).backward()
                    loss_value = window_loss.item()
                    if not math.isfinite(loss_value):
                        raise FloatingPointError(f"Non-finite training loss at step {state['step'] + 1}")
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training["max_grad_norm"])
                    previous_scale = scaler.get_scale()
                    last_lr = optimizer.param_groups[0]["lr"]
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= previous_scale:
                        break
                    restore_rng(retry_rng)
                    log(f"fp16 overflow; retrying update with scale={scaler.get_scale()}")

                scheduler.step()
                state["step"] += 1
                state["next_sample"] = end
                if end == len(train_data):
                    state["epoch"] += 1
                    state["next_sample"] = 0
                state["log_loss_sum"] += loss_value
                state["log_tokens"] += token_count
                state["log_samples"] += len(indices)
                if (logging["log_interval_steps"] and state["step"] % logging["log_interval_steps"] == 0) \
                        or state["step"] == total_steps:
                    log_training()
                if evaluation["eval_interval_steps"] and state["step"] % evaluation["eval_interval_steps"] == 0:
                    run_evaluation()
                if checkpoint_config["save_interval_steps"] and state["step"] % checkpoint_config["save_interval_steps"] == 0:
                    save_checkpoint(f"step_{state['step']:08d}.pt")
                    periodic = sorted(checkpoints.glob("step_*.pt"))
                    keep = checkpoint_config["keep_last_n"]
                    for obsolete in periodic[:-keep] if keep else periodic:
                        obsolete.unlink()

        if evaluation["eval_at_end"]:
            run_evaluation()
        if checkpoint_config["save_last"]:
            save_checkpoint("last.pt")
        plot_losses(state["train_history"], state["eval_history"], output_evaluation / "loss_curve.png")
        log(f"Complete: {state['step']} optimizer steps in {time.monotonic() - started:.1f}s; results: {run_dir}")


def main(argv=None):
    configure_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="training JSON configuration")
    args = parser.parse_args(argv)
    config_path = project_path(args.config)
    with config_path.open(encoding="utf-8") as source:
        config = json.load(source)
    run_dir = project_path(config["experiment"]["output_dir"]) / config["experiment"]["experiment_name"]
    run_dir = run_dir.resolve()
    if not run_dir.is_relative_to(PROJECT_ROOT):
        raise ValueError("The experiment output directory must be inside the project")
    if not config["checkpoint"]["resume_from"] and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Experiment directory is not empty: {run_dir}; set resume_from to resume")
    log_dir = (run_dir / config["logging"]["log_dir"]).resolve()
    if not log_dir.is_relative_to(run_dir):
        raise ValueError("logging.log_dir must stay inside the experiment directory")
    log_dir.mkdir(parents=True, exist_ok=True)
    with TerminalLog(log_dir / "terminal.log", config["logging"]["log_to_console"]):
        try:
            log(f"Training config: {config_path}")
            log(json.dumps(config, ensure_ascii=False, indent=2))
            run_training(config, run_dir, log_dir)
        except (Exception, KeyboardInterrupt) as error:
            traceback.print_exc()
            return 130 if isinstance(error, KeyboardInterrupt) else 1
        finally:
            # Compiler workers inherit stdout/stderr. Close them before waiting
            # for the terminal-log pipe to reach EOF.
            compiler = sys.modules.get("torch._inductor.async_compile")
            if config["system"]["compile_model"] and compiler is not None:
                compiler.shutdown_compile_workers()
    return 0


if __name__ == "__main__":
    sys.exit(main())
