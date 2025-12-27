#!/usr/bin/env python3
import argparse
import pathlib
import queue
import sys
import time

import numpy as np
import sounddevice as sd  # pip install sounddevice

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_MS = 200       # how often to flush audio into the pipe
DTYPE = np.int16

def main():
    parser = argparse.ArgumentParser(description="Stream microphone audio into a FIFO as 16-bit PCM.")
    parser.add_argument("--fifo", required=True, help="Path to the FIFO pipe Whisper reads from")
    parser.add_argument("--device", type=int, help="Optional sounddevice input index")
    parser.add_argument("--sr", type=int, default=SAMPLE_RATE, help="Sample rate for capture (default 16 kHz)")
    parser.add_argument("--chunk-ms", type=int, default=CHUNK_MS, help="Chunk size in milliseconds")
    args = parser.parse_args()

    fifo_path = pathlib.Path(args.fifo)
    if not fifo_path.exists():
        sys.exit(f"FIFO {fifo_path} does not exist (use mkfifo first)")

    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    frames_per_chunk = int(args.sr * args.chunk_ms / 1000)

    with sd.InputStream(
        samplerate=args.sr,
        channels=CHANNELS,
        dtype="float32",
        blocksize=frames_per_chunk,
        callback=callback,
        device=args.device,
    ), fifo_path.open("wb", buffering=0) as fifo:
        print(f"Writing microphone audio to {fifo_path} ({args.sr} Hz, {CHANNELS} ch, {args.chunk_ms} ms chunks)")
        print("Ctrl-C to stop.")
        try:
            while True:
                data = q.get()
                pcm16 = np.clip(data, -1.0, 1.0)
                pcm16 = (pcm16 * np.iinfo(DTYPE).max).astype(DTYPE)
                fifo.write(pcm16.tobytes())
        except KeyboardInterrupt:
            print("\nStopped.")

if __name__ == "__main__":
    main()
