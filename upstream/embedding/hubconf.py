from .expert import UpstreamExpert as _UpstreamExpert


def embedding(ckpt, *args, **kwargs):
    return _UpstreamExpert(ckpt, *args, **kwargs)
