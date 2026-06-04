"""
debug_escuchar.py
=================
Prueba el reconocedor HMM+MFCC+VQ del paquete voice_control
con el micrófono o con un archivo WAV.

Muestra los scores de todos los modelos para debuggear.

Uso desde src/voice_control/:
  python3 debug_escuchar.py              # graba 2.5 s y clasifica
  python3 debug_escuchar.py --dur 3      # graba 3 s
  python3 debug_escuchar.py --loop       # repite hasta Ctrl+C
  python3 debug_escuchar.py --wav /ruta/a/archivo.wav
  python3 debug_escuchar.py --wav /ruta/a/archivo.wav --word stop
"""

import os
import sys
import argparse
import numpy as np
import scipy.io.wavfile as wavfile

# Usar el paquete local directamente (sin necesidad de colcon build)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voice_control.hmm_recognizer import HMMRecognizer
from voice_control.hmm_utils import detect_endpoints, extract_mfcc, quantize, PALABRAS, FS


def cargar_recognizer():
    r = HMMRecognizer()
    if not r.load():
        sys.exit("❌ Modelos no encontrados. Verifica models/hmm_models/")
    print(f"✓ Modelos cargados: {r.vocabulary}")
    return r


def _scores_detallados(signal, fs, recognizer):
    """Devuelve dict {palabra: score} para todos los modelos."""
    start, end = detect_endpoints(signal.astype(np.float32), fs=fs)
    segment = signal[start:end]
    if len(segment) < int(0.1 * fs):
        segment = signal

    mfcc    = extract_mfcc(segment.astype(np.float32), fs=fs)
    indices = quantize(mfcc, recognizer._centroids)

    scores = {}
    for word, hmm in recognizer._hmms.items():
        scores[word] = hmm.log_likelihood(indices)

    dur_total = len(signal) / fs
    dur_seg   = len(segment) / fs
    print(f"\n  Audio total  : {dur_total:.2f} s  |  Segmento voz: {dur_seg:.2f} s  [{start}–{end}]")
    print(f"  Frames MFCC  : {len(mfcc)}  |  Secuencia VQ: {indices[:12].tolist()} ...")
    return scores


def imprimir_scores(scores, real_word=None):
    if not scores:
        print("  ⚠ Sin scores — señal demasiado corta.")
        return None

    best = max(scores, key=lambda w: scores[w])
    best_score = scores[best]

    print()
    print(f"  {'Modelo':<12} {'Log-lik':>10}  {'Δ vs mejor':>12}  Bar")
    print("  " + "─" * 60)
    for w, s in sorted(scores.items(), key=lambda x: -x[1]):
        delta = s - best_score
        bar   = max(0, int(30 + delta * 30 / max(abs(best_score), 1)))
        bar   = min(bar, 30)
        fill  = '█' * bar + '░' * (30 - bar)
        tag   = ''
        if w == best:
            tag = ' ◀ PREDICHO'
        if real_word and w == real_word and w != best:
            tag = ' ← REAL'
        print(f"  {w:<12} {s:>10.1f}  {delta:>+12.1f}  {fill}{tag}")
    print()

    ok = (real_word is None) or (best == real_word)
    mark = '✓' if ok else '✗'
    if real_word:
        print(f"  {mark} REAL: «{real_word}»  →  PREDICHO: «{best}»")
    else:
        print(f"  ══ RESULTADO: «{best.upper()}» ══")

    return best


def grabar(dur_s, fs=FS):
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("❌ Instala sounddevice:  pip install sounddevice")
    print(f"\n  🎙  Grabando {dur_s:.1f} s — ¡habla ahora! ...", end='', flush=True)
    audio = sd.rec(int(dur_s * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("  [ok]")
    return audio.flatten()


def clasificar_wav(path, recognizer, real_word=None):
    rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    signal = data.astype(np.float32)
    if signal.max() > 1.5:
        signal /= 32768.0
    if rate != FS:
        print(f"  ⚠ WAV a {rate} Hz, sistema espera {FS} Hz — puede afectar resultado")
    scores = _scores_detallados(signal, rate, recognizer)
    return imprimir_scores(scores, real_word)


def main():
    parser = argparse.ArgumentParser(description='Debug reconocedor HMM voz')
    parser.add_argument('--dur',  type=float, default=2.5,
                        help='Duración de grabación en segundos (default: 2.5)')
    parser.add_argument('--loop', action='store_true',
                        help='Repetir hasta Ctrl+C')
    parser.add_argument('--wav',  default=None,
                        help='Archivo WAV a clasificar (no graba)')
    parser.add_argument('--word', default=None,
                        help='Palabra real para comparar contra la predicha')
    args = parser.parse_args()

    print("=" * 60)
    print("  Debug reconocedor HMM + MFCC + VQ")
    print(f"  Vocabulario: {', '.join(PALABRAS)}")
    print("=" * 60)

    recognizer = cargar_recognizer()

    # ── Modo WAV ──────────────────────────────────────────────────
    if args.wav:
        print(f"\nClasificando: {args.wav}")
        clasificar_wav(args.wav, recognizer, args.word)
        return

    # ── Modo batch: probar todos los WAV de un directorio ─────────
    # Ejemplo: python debug_escuchar.py --wav /ruta/voz/stop
    # (No implementado aquí; usa --wav con archivo individual)

    # ── Modo grabación ────────────────────────────────────────────
    n = 0
    while True:
        n += 1
        if args.loop:
            print(f"\n─── Intento #{n} ───")

        try:
            signal = grabar(args.dur)
        except KeyboardInterrupt:
            print("\nSaliendo.")
            break

        scores = _scores_detallados(signal, FS, recognizer)
        imprimir_scores(scores)

        if not args.loop:
            break

        try:
            input("  Presiona ENTER para grabar de nuevo (Ctrl+C para salir)...")
        except KeyboardInterrupt:
            print("\nSaliendo.")
            break


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nSaliendo.")
