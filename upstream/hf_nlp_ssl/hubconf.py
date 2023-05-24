from .expert import UpstreamExpert as _UpstreamExpert


def hf_nlp_ssl(ckpt, *args, **kwargs):
    return _UpstreamExpert(ckpt, *args, **kwargs)
