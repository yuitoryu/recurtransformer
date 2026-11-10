"""Measure training VRAM with synthetic full-context inputs.

python -B -m src.test_vram --config config/example --compile off
Overrides do not modify config files. No dataset or checkpoint is loaded.
"""

import argparse
import json
import math
from pathlib import Path
import sys

from .train import (configure_environment, project_path, summed_loss,
                    compile_for_training, capture_rng, restore_rng)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int, default=3,
                        help="successful updates after the first update (default: 3)")
    parser.add_argument("--compile", choices=("off", "full", "regional"))
    args = parser.parse_args(argv)
    configure_environment()

    import torch
    from . import model as model_module

    stage = "configuration"
    try:
        directory = project_path(args.config)
        with (directory / "train.json").open(encoding="utf-8") as source:
            config = json.load(source)
        with (directory / "model.json").open(encoding="utf-8") as source:
            model_config = json.load(source)
        training, system = config["training"], config["system"]
        for key, value in (("micro_batch_size", args.batch_size),
                           ("gradient_accumulation_steps", args.grad_accum),
                           ("precision", args.precision)):
            if value is not None:
                training[key] = value
        if args.compile is not None:
            system["compile_model"] = args.compile != "off"
            system["compile_mode"] = args.compile
        batch = training["micro_batch_size"]
        accumulation = training["gradient_accumulation_steps"]
        precision = training["precision"]
        device = torch.device(args.device or system["device"])
        if min(batch, accumulation, args.steps) < 1:
            raise ValueError("batch-size, grad-accum and steps must be positive")
        if precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32, bf16 or fp16")
        if training["optimizer"].lower() != "adamw":
            raise ValueError("Only optimizer='adamw' is supported")
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("VRAM measurement requires an available CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("The requested CUDA device does not support bf16")
        torch.manual_seed(config["experiment"]["seed"])
        compile_mode = system.get("compile_mode", "full") if system["compile_model"] else "off"
        print(f"Model: {model_config['use_model']}; device={device}; "
              f"micro_batch={batch}; grad_accum={accumulation}; precision={precision}; "
              f"compile={compile_mode}; measured_steps={args.steps}", flush=True)
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name}; total={props.total_memory / 1024**3:.3f} GiB", flush=True)
        model_args = {k: v for k, v in model_config.items() if k not in
                      {"name", "use_model", "use_weighted_from_each_loop", "use_regularizer"}}
        model_args["device"] = "cpu"
        stage = "model construction"
        model = getattr(model_module, model_config["use_model"])(**model_args)
        length = model.max_seq_len if args.seq_len is None else args.seq_len
        if not 1 <= length <= model.max_seq_len:
            raise ValueError(f"seq-len must be between 1 and max_seq_len={model.max_seq_len}")
        print(f"Input: ({batch}, {length}); context_window={model.max_seq_len}; "
              f"parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
        stage = "model transfer to CUDA"
        model = model.to(device).train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=training["learning_rate"],
            betas=(training["beta1"], training["beta2"]),
            eps=training["adam_eps"], weight_decay=training["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
        stage = "compilation setup"
        trained_model = compile_for_training(model, system)
        # CPU inputs mirror the training loop's copies and avoid persistent GPU batches.
        tokens = torch.randint(model.vocab_size, (batch, length + 1))
        inputs, targets = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()

        def update(label):
            nonlocal stage
            retry_rng = capture_rng() if scaler.is_enabled() else None
            while True:
                optimizer.zero_grad(set_to_none=True)
                window_loss = torch.zeros((), device=device, dtype=torch.float64)
                for micro in range(accumulation):
                    stage = f"{label}: forward {micro + 1}/{accumulation}"
                    loss = summed_loss(trained_model, inputs, targets, device, precision)
                    window_loss += loss.detach().double()
                    stage = f"{label}: backward {micro + 1}/{accumulation}"
                    scaler.scale(loss / (batch * length * accumulation)).backward()
                if not math.isfinite(window_loss.item()):
                    raise FloatingPointError("Non-finite training loss")
                stage = f"{label}: gradient clipping"
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["max_grad_norm"])
                stage = f"{label}: optimizer step"
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    break
                restore_rng(retry_rng)
                print(f"fp16 overflow; retrying with scale={scaler.get_scale()}", flush=True)

        peaks = []
        for label, count in (("First update", 1), ("Subsequent updates", args.steps)):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            for index in range(count):
                update(f"{label} {index + 1}/{count}")
            torch.cuda.synchronize(device)
            peak = (torch.cuda.max_memory_allocated(device),
                    torch.cuda.max_memory_reserved(device))
            peaks.append(peak)
            print(f"{label}: allocated={peak[0] / 1024**3:.3f} GiB; "
                  f"reserved={peak[1] / 1024**3:.3f} GiB", flush=True)
        print(f"Overall: allocated={max(p[0] for p in peaks) / 1024**3:.3f} GiB; "
              f"reserved={max(p[1] for p in peaks) / 1024**3:.3f} GiB")
        print(f"AdamW state initialized for {len(optimizer.state)} parameters.")
        print("PyTorch allocator metrics only, not total GPU usage. "
              "Allocated and reserved must not be added together.")
        return 0
    except torch.cuda.OutOfMemoryError as error:
        print(f"CUDA OOM during {stage}: {error}", file=sys.stderr, flush=True)
        return 1
    except (ValueError, RuntimeError, FloatingPointError) as error:
        print(f"Failed during {stage}: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        compiler = sys.modules.get("torch._inductor.async_compile")
        if compiler is not None:
            compiler.shutdown_compile_workers()


if __name__ == "__main__":
    sys.exit(main())
