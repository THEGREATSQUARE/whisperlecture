import io
import os
import tempfile
import logging
import shutil
from pathlib import Path
from typing import Tuple
from datetime import datetime

from flask import Flask, jsonify, request, send_file
from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('whisper_server.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

load_dotenv()
client = OpenAI()

# OpenAI Whisper API limits
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
CHUNK_DURATION = 10 * 60  # 10 minutes in seconds

# FFmpeg configuration - you can set this in your .env file
FFMPEG_PATH = os.getenv('FFMPEG_PATH', 'ffmpeg')  # Default to 'ffmpeg' (PATH lookup)


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available in PATH or specified path"""
    global FFMPEG_PATH
    
    if FFMPEG_PATH != 'ffmpeg':
        # Check if custom path exists
        if os.path.exists(FFMPEG_PATH):
            logger.info(f"Using custom FFmpeg path: {FFMPEG_PATH}")
            return True
    
    # Check PATH
    if shutil.which('ffmpeg') is not None:
        logger.info("FFmpeg found in PATH")
        return True
    
    # Common Windows FFmpeg locations
    common_paths = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        os.path.expanduser(r"~\ffmpeg\bin\ffmpeg.exe"),
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            logger.info(f"FFmpeg found at: {path}")
            FFMPEG_PATH = path
            return True
    
    return False


def get_ffmpeg_command() -> list[str]:
    """Get the ffmpeg command to use"""
    if FFMPEG_PATH == 'ffmpeg':
        return ['ffmpeg']
    else:
        return [FFMPEG_PATH]


def log_request_info(file_size: int, filename: str):
    """Log detailed request information"""
    size_mb = file_size / (1024 * 1024)
    logger.info(f"Processing file: {filename}")
    logger.info(f"File size: {size_mb:.2f} MB")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if file_size > MAX_FILE_SIZE:
        logger.warning(f"File exceeds OpenAI limit ({MAX_FILE_SIZE / (1024*1024):.1f}MB). Will chunk into segments.")
        if not check_ffmpeg_available():
            logger.error("FFmpeg not found! Large files cannot be processed without FFmpeg.")


def chunk_audio_file(input_path: Path, chunk_duration: int = CHUNK_DURATION) -> tuple[list[Path], Path]:
    """Split large audio file into chunks using ffmpeg"""
    if not check_ffmpeg_available():
        raise RuntimeError("FFmpeg not found in PATH. Please install FFmpeg to process large files.")
    
    logger.info("Using ffmpeg for audio chunking")
    return chunk_with_ffmpeg(input_path, chunk_duration)


def chunk_with_ffmpeg(input_path: Path, chunk_duration: int) -> tuple[list[Path], Path]:
    """Chunk audio using ffmpeg command line"""
    import subprocess
    
    temp_dir = Path(tempfile.mkdtemp())
    chunks = []
    ffmpeg_cmd = get_ffmpeg_command()
    
    # Get duration first
    logger.info("Getting audio duration...")
    result = subprocess.run([
        *ffmpeg_cmd, '-i', str(input_path), '-f', 'null', '-'
    ], capture_output=True, text=True)
    
    # Extract duration from stderr (ffmpeg outputs info there)
    duration_seconds = None
    for line in result.stderr.split('\n'):
        if 'Duration:' in line:
            logger.info(f"Audio duration: {line.strip()}")
            # Parse duration: Duration: 00:40:16.00
            try:
                time_part = line.split('Duration:')[1].split(',')[0].strip()
                h, m, s = map(float, time_part.split(':'))
                duration_seconds = int(h * 3600 + m * 60 + s)
                logger.info(f"Parsed duration: {duration_seconds} seconds ({duration_seconds//60} minutes)")
            except Exception as e:
                logger.warning(f"Could not parse duration: {e}")
            break
    
    if duration_seconds is None:
        logger.warning("Could not determine audio duration, will use default chunking")
        duration_seconds = 1000  # fallback
    
    # Calculate how many chunks we actually need
    total_chunks = (duration_seconds + chunk_duration - 1) // chunk_duration
    logger.info(f"Will create {total_chunks} chunks of {chunk_duration//60} minutes each")
    
    # Create chunks
    for i in range(total_chunks):
        start_time = i * chunk_duration
        end_time = min(start_time + chunk_duration, duration_seconds)
        
        # Skip if this chunk would be beyond the audio
        if start_time >= duration_seconds:
            logger.info(f"Skipping chunk {i+1} - beyond audio duration")
            break
            
        chunk_path = temp_dir / f"chunk_{i:03d}.mp3"
        
        # Adjust chunk duration for the last chunk
        actual_chunk_duration = end_time - start_time
        
        cmd = [
            *ffmpeg_cmd, '-y', '-i', str(input_path),
            '-ss', str(start_time), '-t', str(actual_chunk_duration),
            '-c:a', 'libmp3lame', '-b:a', '128k', str(chunk_path)
        ]
        
        logger.info(f"Creating chunk {i+1}/{total_chunks}: {start_time//60}:{start_time%60:02d} - {end_time//60}:{end_time%60:02d}")
        result = subprocess.run(cmd, capture_output=True)
        
        if result.returncode != 0:
            logger.warning(f"ffmpeg chunk {i+1} failed: {result.stderr}")
            break
            
        if chunk_path.exists() and chunk_path.stat().st_size > 0:
            chunks.append(chunk_path)
            logger.info(f"Successfully created chunk {i+1}: {chunk_path.name}")
        else:
            logger.info(f"Chunk {i+1} is empty or failed, stopping")
            break
    
    if not chunks:
        raise RuntimeError("Failed to create any audio chunks")
    
    logger.info(f"Successfully created {len(chunks)} chunks")
    return chunks, temp_dir


def transcribe_to_srt(temp_path: Path) -> str:
    """Transcribe a single audio file to SRT format"""
    file_size = temp_path.stat().st_size
    size_mb = file_size / (1024 * 1024)
    
    logger.info(f"Transcribing file: {temp_path.name} ({size_mb:.2f} MB)")
    
    with temp_path.open("rb") as f:
        srt_text = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="srt",
        )
    
    logger.info(f"Transcription completed for {temp_path.name}")
    return srt_text  # type: ignore[return-value]


def transcribe_large_file(input_path: Path) -> str:
    """Handle large files by chunking and transcribing separately"""
    logger.info(f"Large file detected, chunking into {CHUNK_DURATION//60} minute segments")
    
    chunks, temp_dir = chunk_audio_file(input_path)
    all_transcripts = []
    
    try:
        for i, chunk_path in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}: {chunk_path.name}")
            
            try:
                srt_text = transcribe_to_srt(chunk_path)
                # Adjust timestamps for each chunk
                adjusted_srt = adjust_srt_timestamps(srt_text, i * CHUNK_DURATION)
                all_transcripts.append(adjusted_srt)
                
            except Exception as e:
                logger.error(f"Failed to transcribe chunk {i+1}: {e}")
                # Continue with other chunks
                continue
                
    finally:
        # Clean up temporary chunks
        for chunk_path in chunks:
            try:
                chunk_path.unlink()
            except:
                pass
        try:
            temp_dir.rmdir()
        except:
            pass
    
    # Combine all transcripts
    combined_srt = combine_srt_transcripts(all_transcripts)
    logger.info(f"Combined {len(chunks)} chunks into final transcript")
    
    return combined_srt


def adjust_srt_timestamps(srt_text: str, offset_seconds: int) -> str:
    """Adjust SRT timestamps by adding offset"""
    lines = srt_text.split('\n')
    adjusted_lines = []
    
    for line in lines:
        if ' --> ' in line:
            # Parse and adjust timestamp
            start, end = line.split(' --> ')
            start_adj = adjust_timestamp(start, offset_seconds)
            end_adj = adjust_timestamp(end, offset_seconds)
            adjusted_lines.append(f"{start_adj} --> {end_adj}")
        else:
            adjusted_lines.append(line)
    
    return '\n'.join(adjusted_lines)


def adjust_timestamp(timestamp: str, offset_seconds: int) -> str:
    """Adjust a single timestamp by adding offset"""
    try:
        # Parse HH:MM:SS,mmm format
        time_part, ms_part = timestamp.split(',')
        h, m, s = map(int, time_part.split(':'))
        ms = int(ms_part)
        
        total_seconds = h * 3600 + m * 60 + s + offset_seconds
        
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    except:
        return timestamp


def combine_srt_transcripts(transcripts: list[str]) -> str:
    """Combine multiple SRT transcripts into one"""
    combined = []
    subtitle_index = 1
    
    for transcript in transcripts:
        lines = transcript.strip().split('\n')
        current_block = []
        
        for line in lines:
            if line.strip() == '':
                if current_block:
                    # Process completed block
                    if len(current_block) >= 2:
                        # Replace subtitle index
                        combined.append(str(subtitle_index))
                        subtitle_index += 1
                        # Add timestamp and text
                        combined.extend(current_block[1:])
                        combined.append('')
                    current_block = []
            else:
                current_block.append(line)
        
        # Handle last block
        if current_block and len(current_block) >= 2:
            combined.append(str(subtitle_index))
            subtitle_index += 1
            combined.extend(current_block[1:])
            combined.append('')
    
    return '\n'.join(combined)


def srt_to_structured_text(srt_text: str) -> str:
    """Convert SRT to structured text with timestamps"""
    out_lines = []
    block = []
    
    for line in srt_text.splitlines():
        if line.strip() == "":
            if block:
                if len(block) >= 2:
                    ts = block[1].strip()
                    text = " ".join(s.strip() for s in block[2:]).strip()
                    out_lines.append(f"[{ts}]")
                    if text:
                        out_lines.append(text)
                    out_lines.append("")
                block = []
            continue
        block.append(line)
    
    if block:
        if len(block) >= 2:
            ts = block[1].strip()
            text = " ".join(s.strip() for s in block[2:]).strip()
            out_lines.append(f"[{ts}]")
            if text:
                out_lines.append(text)
            out_lines.append("")
    
    return "\n".join(out_lines).strip() + "\n"


@app.post("/api/transcribe")
def upload_and_transcribe():
    """Handle file upload and transcription"""
    if "file" not in request.files:
        logger.error("No file part in request")
        return jsonify({"error": "No file part"}), 400
    
    f = request.files["file"]
    if f.filename == "":
        logger.error("No selected file")
        return jsonify({"error": "No selected file"}), 400

    # Log request details
    file_size = 0
    suffix = Path(f.filename).suffix.lower() or ".wav"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        f.save(tmp.name)
        tmp_path = Path(tmp.name)
        file_size = tmp_path.stat().st_size

    log_request_info(file_size, f.filename)

    try:
        # Choose transcription method based on file size
        if file_size > MAX_FILE_SIZE:
            if not check_ffmpeg_available():
                return jsonify({
                    "error": "File too large and FFmpeg not available. Please install FFmpeg to process files over 25MB."
                }), 400
            
            logger.info("Using chunked transcription for large file")
            srt_text = transcribe_large_file(tmp_path)
        else:
            logger.info("Using direct transcription for small file")
            srt_text = transcribe_to_srt(tmp_path)
        
        # Convert to structured text
        structured = srt_to_structured_text(srt_text)
        out_bytes = structured.encode("utf-8")
        
        logger.info(f"Transcription completed successfully. Output size: {len(out_bytes)} bytes")
        
        return send_file(
            io.BytesIO(out_bytes),
            mimetype="text/plain; charset=utf-8",
            as_attachment=True,
            download_name=f"{Path(f.filename).stem}_transcript.txt",
        )
        
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500
        
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.get("/")
def index():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    app.static_folder = "static"
    logger.info("Starting WhisperLecture server...")
    logger.info(f"Max file size: {MAX_FILE_SIZE / (1024*1024):.1f} MB")
    logger.info(f"Chunk duration: {CHUNK_DURATION//60} minutes")
    
    if check_ffmpeg_available():
        logger.info(f"FFmpeg found - large file support enabled (using: {FFMPEG_PATH})")
    else:
        logger.warning("FFmpeg not found - only files under 25MB can be processed")
        logger.info("To enable large file support, either:")
        logger.info("1. Add FFmpeg to your PATH, or")
        logger.info("2. Set FFMPEG_PATH in your .env file, or")
        logger.info("3. Install FFmpeg to a common location (C:\\ffmpeg\\bin\\ffmpeg.exe)")
    
    app.run(host="127.0.0.1", port=5000, debug=True)
