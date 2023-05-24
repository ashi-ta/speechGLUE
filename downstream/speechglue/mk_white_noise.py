import os
import sys

import numpy as np
import soundfile

SAMPLE_RATE = 16000

np.random.seed(seed=0)

if len(sys.argv) == 1:
    duration_msec = 50
    use_50ms = True
else:
    duration_msec = int(sys.argv[1])
    use_50ms = False

duration = int(SAMPLE_RATE * duration_msec / 1000)
sep_sig = np.random.randn(duration)
# prevent from a saturation
sep_sig = sep_sig / np.max(np.abs(sep_sig)) * 0.99

if use_50ms:
    out_path = os.path.join("dump", "white_noise.wav")
else:
    out_path = os.path.join("dump", f"white_noise_{duration_msec}ms.wav")
soundfile.write(out_path, sep_sig, SAMPLE_RATE, "PCM_16")
