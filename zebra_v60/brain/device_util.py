import torch


def get_device(preferred="auto"):
    """Return the best available torch device.

    preferred: "auto" (CUDA > MPS > CPU), "cuda", "mps", "cpu"
    """
    if preferred == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return preferred
