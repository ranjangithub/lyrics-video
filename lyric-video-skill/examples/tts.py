"""Tiny espeak-ng TTS via ctypes (test-only helper to fake a vocal track)."""
import ctypes, numpy as np, espeakng_loader as el

lib = ctypes.cdll.LoadLibrary(el.get_library_path())
SR = None
_buf = []
CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p)

@CB
def _cb(wav, n, events):
    if n > 0:
        _buf.append(np.ctypeslib.as_array(wav, shape=(n,)).copy())
    return 0

def init(voice=b"hi", rate=150, pitch=40):
    global SR
    SR = lib.espeak_Initialize(1, 0, el.get_data_path().encode(), 0)  # AUDIO_OUTPUT_RETRIEVAL
    lib.espeak_SetSynthCallback(_cb)
    lib.espeak_SetVoiceByName(voice)
    lib.espeak_SetParameter(1, rate, 0)   # espeakRATE
    lib.espeak_SetParameter(3, pitch, 0)  # espeakPITCH
    return SR

def say(text):
    _buf.clear()
    b = text.encode("utf-8")
    lib.espeak_Synth(b, len(b) + 1, 0, 0, 0, 0x10, None, None)  # espeakCHARS_UTF8
    lib.espeak_Synchronize()
    return (np.concatenate(_buf) if _buf else np.zeros(1, np.int16)).astype(np.float32) / 32768.0
