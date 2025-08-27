import os
from dotenv import load_dotenv

print("Before load_dotenv():")
print(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY', 'NOT SET')}")
print(f"FFMPEG_PATH: {os.getenv('FFMPEG_PATH', 'NOT SET')}")

print("\nLoading .env file...")
load_dotenv()

print("\nAfter load_dotenv():")
print(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY', 'NOT SET')}")
print(f"FFMPEG_PATH: {os.getenv('FFMPEG_PATH', 'NOT SET')}")

print(f"\nCurrent working directory: {os.getcwd()}")
print(f".env file exists: {os.path.exists('.env')}")
