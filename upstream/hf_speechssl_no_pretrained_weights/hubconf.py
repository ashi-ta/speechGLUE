from .expert import UpstreamExpert as _UpstreamExpert


def hf_speechssl_no_pretrained_weights(ckpt, *args, **kwargs):
    return _UpstreamExpert(ckpt, *args, **kwargs)
