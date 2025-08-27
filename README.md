### WhisperLecture — Transcribe lectures with OpenAI Whisper API

**What this is**: A minimal Python tool to transcribe audio files (single file or a folder) using OpenAI's Whisper API, saving `.txt`, `.srt`, or `.vtt` outputs. Now with web UI, progress tracking, and large file support!

### Prerequisites
- Python 3.9+
- An OpenAI API key with access to `whisper-1`
- **FFmpeg** (required for large file chunking) - [Download here](https://ffmpeg.org/download.html)

### Install
```powershell
# In PowerShell, from the project folder
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Important**: For large file support (>25MB), you MUST install FFmpeg:
1. Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)
2. Extract to a folder (e.g., `C:\ffmpeg`)
3. Add the `bin` folder to your system PATH
4. Restart your terminal/PowerShell

**Note**: The script will automatically chunk files larger than 25MB into 10-minute segments using FFmpeg.

### Configure your API key (Windows PowerShell)
- One-time (persist for your user):
```powershell
[System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "YOUR_KEY_HERE", "User")
# Restart terminal to pick up the change
```
- Or for current session only:
```powershell
$env:OPENAI_API_KEY = "YOUR_KEY_HERE"
```

### Usage (CLI)
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
- Files over 25MB are automatically split into 10-minute chunks
- Each chunk is transcribed separately
- Timestamps are adjusted and combined into a single transcript
- Progress is logged to both terminal and `whisper_server.log`
- **Requires FFmpeg to be installed and in PATH**

### Notes
- The script uses the OpenAI Python SDK (`openai` v1). Model: `whisper-1`.
- API returns plain text for `--format text`, `.srt` for subtitles, or `.vtt`.
- The web UI always returns `.txt` using SRT-to-text conversion.
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
