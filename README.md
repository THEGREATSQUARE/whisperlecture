### WhisperAudio — Fast, local lecture transcription (API optional)

**What this is**: A simple tool with a drag‑and‑drop web UI to transcribe audio into clean text (`.txt`, `.srt`, `.vtt`). It prioritizes running locally with `faster-whisper` for speed, quality, privacy, and reliability. An OpenAI API fallback is available but not recommended.

### Prerequisites
- Python 3.9+
- **FFmpeg** (required for large file chunking) — see install below
- (Optional) OpenAI API key — only if you want the API fallback

### Install (Windows PowerShell)
```powershell
# In PowerShell, from the project folder
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**FFmpeg (required for large files >25MB)**
1. Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)
2. Extract to a folder (e.g., `C:\ffmpeg`)
3. Add the `bin` folder to your system PATH
4. Restart your terminal/PowerShell

Note: The server automatically chunks files larger than 25MB into 5‑minute segments using FFmpeg.

If FFmpeg is not on PATH, set it explicitly:
```powershell
$env:FFMPEG_PATH = "C:\\ffmpeg\\bin\\ffmpeg.exe"
```

### Quick Start — Local (recommended)
```powershell
# 1) Activate the venv (from project root)
.\.venv\Scripts\Activate.ps1

# 2) Enable local faster-whisper
$env:USE_LOCAL_WHISPER = "1"
$env:LOCAL_MODEL = "small.en"      # good default; try base|small|medium|large
$env:LOCAL_COMPUTE = "int8"        # int8|int8_float32|float16|float32

# 3) (Optional) Keep VAD disabled for most consistent transcripts
$env:VAD_FILTER = "0"

# 4) Start the web UI
python server.py

# 5) Open http://127.0.0.1:5000 and drop audio files
```

### CLI Usage (local)
```powershell
# Transcribe a single file (default output: .txt next to the audio)
python transcribe.py --input "C:\path\to\lecture.mp3"

# Transcribe a folder (all supported audio files, recursively)
python transcribe.py --input "C:\path\to\folder"

# Choose output format: text|srt|vtt (default: text)
python transcribe.py --input "C:\path\to\lecture.wav" --format srt

# Send outputs to a specific directory
python transcribe.py --input "C:\path\to\folder" --outdir "C:\path\to\transcripts"

# Control concurrency (parallel uploads)
python transcribe.py --input "C:\path\to\folder" --workers 3
```

### Local Web UI (drag-and-drop with progress tracking)
```powershell
# Start the local server
python server.py
# Then open http://127.0.0.1:5000 in your browser
```

**Features:**
- Drag-and-drop audio files
- Progress bar with time estimates
- Automatic large file chunking (>25MB)
- Real-time logging in terminal
- Structured text output with timestamps

### Supported audio formats
Extensions: `.mp3 .mp4 .mpeg .mpga .m4a .wav .webm .ogg .flac .wma .aac .mkv .opus`

### Large file handling
- Files over 25MB are automatically split into 5-minute chunks
- Each chunk is transcribed separately
- Timestamps are adjusted and combined into a single transcript
- Progress is logged to both terminal and `whisper_server.log`
- **Requires FFmpeg to be installed and in PATH**

### Local faster-whisper options
Local transcription typically yields better control and quality than the legacy `whisper-1` API model, and it keeps audio on your machine.

Enable local mode and choose a model:
```powershell
$env:USE_LOCAL_WHISPER = "1"         # enable local faster-whisper
$env:LOCAL_MODEL = "small.en"        # e.g. tiny|base|small|medium|large, *.en variants
$env:LOCAL_COMPUTE = "int8"          # int8|int8_float32|float16|float32
python server.py
```

Voice Activity Detection (VAD) for local mode:
- By default, VAD is disabled for maximum completeness/consistency of transcripts.
- You can enable and tune it to reduce long silences or strong background noise.

Disable VAD (recommended for most consistent output):
```powershell
$env:USE_LOCAL_WHISPER = "1"
$env:VAD_FILTER = "0"
python server.py
```

Enable VAD (lenient settings to avoid clipping speech):
```powershell
$env:USE_LOCAL_WHISPER = "1"
$env:VAD_FILTER = "1"
$env:VAD_MIN_SILENCE_MS = "800"   # 600–1200 good range
$env:VAD_MIN_SPEECH_MS = "150"    # 100–200
$env:VAD_SPEECH_PAD_MS = "400"    # 300–600
$env:VAD_THRESHOLD = "0.4"        # 0.3–0.5
python server.py
```

Notes:
- VAD only affects local faster-whisper. The OpenAI API path does not use VAD.
- If FFmpeg is not in PATH, set `FFMPEG_PATH` as shown above.

### OpenAI API fallback (last resort)
Use this only if you cannot run locally. It uses `whisper-1` and requires an API key.

Configure your API key (one-time, persists for your user):
```powershell
[System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "YOUR_KEY_HERE", "User")
# Restart terminal to pick up the change
```
Or for current session only:
```powershell
$env:OPENAI_API_KEY = "YOUR_KEY_HERE"
```

Then simply do NOT set `USE_LOCAL_WHISPER`, start the server, and use the web UI:
```powershell
python server.py
```

### Notes
- Web UI returns `.txt` using SRT-to-text conversion; CLI supports `.txt | .srt | .vtt`.
- Large files are chunked using FFmpeg and processed sequentially.

### Troubleshooting
- If you see authentication errors, confirm `OPENAI_API_KEY` is available in the environment:
```powershell
$env:OPENAI_API_KEY
```
- If your terminal cannot activate the venv, enable script execution in PowerShell (admin):
```powershell
Set-ExecutionPolicy RemoteSigned
```
- **For large file issues**: Ensure FFmpeg is installed and in your PATH:
```powershell
ffmpeg -version
```
- Check `whisper_server.log` for detailed error information
- The server logs all operations to both console and log file
- If you see "FFmpeg not found" errors, reinstall FFmpeg and restart your terminal

