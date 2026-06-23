import torch
from safetensors import safe_open


def load_safetensor_state_dict(checkpoint_path):
    state_dict = {}
    with safe_open(checkpoint_path, framework="pt") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict


def safe_load_model(model, checkpoint_path, ignore_keys=None, partial_load=None, verbose=True):
    ignore_keys = ignore_keys or []
    partial_load = partial_load or {}

    state_dict = load_safetensor_state_dict(checkpoint_path)
    model_state = model.state_dict()
    filtered_state_dict = {}

    for key, value in state_dict.items():
        if any(ignore_key in key for ignore_key in ignore_keys):
            if verbose:
                print(f"[Skip by rule] {key}")
            continue

        if key in model_state and value.shape == model_state[key].shape:
            filtered_state_dict[key] = value
        elif verbose and key not in partial_load:
            expected_shape = model_state[key].shape if key in model_state else None
            print(f"[Skip mismatch] {key}: ckpt={value.shape}, model={expected_shape}")

    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)

    if partial_load:
        named_params = dict(model.named_parameters())
        named_buffers = dict(model.named_buffers())

        for key, n_rows in partial_load.items():
            if any(ignore_key in key for ignore_key in ignore_keys):
                if verbose:
                    print(f"[Partial skip by rule] {key}")
                continue

            if key not in state_dict or key not in model_state:
                if verbose:
                    print(f"[Partial skip] {key}: not found in checkpoint or model")
                continue

            ckpt_tensor = state_dict[key]
            model_tensor = model_state[key]

            if ckpt_tensor.ndim != model_tensor.ndim:
                if verbose:
                    print(f"[Partial skip] {key}: ndim mismatch ckpt={ckpt_tensor.ndim}, model={model_tensor.ndim}")
                continue
            if ckpt_tensor.ndim < 1:
                if verbose:
                    print(f"[Partial skip] {key}: invalid tensor ndim")
                continue
            if ckpt_tensor.shape[1:] != model_tensor.shape[1:]:
                if verbose:
                    print(
                        f"[Partial skip] {key}: tail dims mismatch "
                        f"ckpt={tuple(ckpt_tensor.shape)}, model={tuple(model_tensor.shape)}"
                    )
                continue

            ckpt_rows = ckpt_tensor.shape[0]
            model_rows = model_tensor.shape[0]
            rows_to_copy = min(ckpt_rows, model_rows) if n_rows is None else min(int(n_rows), ckpt_rows, model_rows)

            with torch.no_grad():
                if key in named_params:
                    target = named_params[key]
                    target[:rows_to_copy].copy_(ckpt_tensor[:rows_to_copy].to(device=target.device, dtype=target.dtype))
                elif key in named_buffers:
                    target = named_buffers[key]
                    target[:rows_to_copy].copy_(ckpt_tensor[:rows_to_copy].to(device=target.device, dtype=target.dtype))
                else:
                    if verbose:
                        print(f"[Partial skip] {key}: not a parameter/buffer")
                    continue

            if verbose:
                print(f"[Partial load] {key}: copied {rows_to_copy}/{model_rows} rows (ckpt={ckpt_rows}, model={model_rows})")

    if verbose:
        print(f"\nMissing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")
        print(f"Successfully loaded {len(filtered_state_dict)} parameters.")
        if partial_load:
            print(f"Partial loaded keys: {list(partial_load.keys())}")

    return model
