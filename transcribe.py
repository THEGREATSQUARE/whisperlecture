import argparse
import concurrent.futures
import mimetypes
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

from openai import OpenAI


SUPPORTED_EXTENSIONS = {
    ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg", ".flac",
    ".wma", ".aac", ".mkv", ".opus",
}


def discover_audio_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return files


def get_output_path(input_path: Path, outdir: Optional[Path], fmt: str) -> Path:
    base_name = input_path.stem
    suffix = {
        "text": ".txt",
        "srt": ".srt",
        "vtt": ".vtt",
    }[fmt]
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir / f"{base_name}{suffix}"
    return input_path.with_suffix(suffix)


def transcribe_file(client: OpenAI, input_path: Path, fmt: str) -> str:
    mime, _ = mimetypes.guess_type(str(input_path))
    with input_path.open("rb") as f:
        if fmt == "text":
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
            )
            return resp.text  # type: ignore[attr-defined]
        else:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format=fmt,
            )
            # For srt/vtt, SDK returns plain string
            return resp  # type: ignore[return-value]


def worker_task(client: OpenAI, src: Path, fmt: str, outdir: Optional[Path]) -> Tuple[Path, Optional[Exception]]:
    try:
        text = transcribe_file(client, src, fmt)
        out_path = get_output_path(src, outdir, fmt)
        out_path.write_text(text, encoding="utf-8")
        return out_path, None
    except Exception as e:
        return src, e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio using OpenAI Whisper API")
    parser.add_argument("--input", required=True, help="Path to audio file or directory")
    parser.add_argument("--format", choices=["text", "srt", "vtt"], default="text", help="Output format")
    parser.add_argument("--outdir", default=None, help="Directory to write outputs (defaults beside inputs)")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel transcriptions for folders")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    client = OpenAI()

    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir) if args.outdir else None

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    targets = discover_audio_files(input_path)
    if not targets:
        print("No audio files found.")
        return

    if len(targets) == 1:
        tgt = targets[0]
        print(f"Transcribing: {tgt}")
        out_path, err = worker_task(client, tgt, args.format, outdir)
        if err:
            raise err
        print(f"Saved: {out_path}")
        return

    print(f"Found {len(targets)} files. Starting with {args.workers} workers...")
    errors: List[Tuple[Path, Exception]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(worker_task, client, p, args.format, outdir)
            for p in targets
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            out_path, err = future.result()
            if err:
                errors.append((out_path, err))

    if errors:
        print("Some files failed:")
        for p, e in errors:
            print(f" - {p}: {e}")
    else:
        print("All files transcribed successfully.")


if __name__ == "__main__":
    main()
