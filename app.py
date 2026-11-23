import os
import uuid
import asyncio
import subprocess
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, Field

import edge_tts
from google import genai
from google.genai import types
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

mcp = MCPServer(
    "iPadAutoVideoMCP",
    stateless_http=True,
    streamable_http_path="/",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)



BASE_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = BASE_DIR / "outputs"
TEMP_DIR = BASE_DIR / "temp"
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8000").rstrip("/")

ai_client = genai.Client(api_key=GEMINI_API_KEY)
mcp = MCPServer("iPadAutoVideoMCP", stateless_http=True, streamable_http_path="/")


app = FastAPI(title="My Video Studio MCP")

app.mount("/files", StaticFiles(directory=str(OUTPUT_DIR)), name="files")

JOBS: Dict[str, Dict[str, Any]] = {}


class SceneInput(BaseModel):
    narration: str = Field(..., description="บทพูดภาษาไทยของฉากนี้ (สั้นกระชับ 1-2 ประโยค)")
    visual_prompt: str = Field(..., description="Prompt ภาษาอังกฤษสำหรับสร้างภาพประกอบฉากนี้ (ไม่ใส่ตัวหนังสือ)")


def get_audio_duration(audio_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return float(res.stdout.strip())


def format_srt_time(seconds: float) -> str:
    millis = int((seconds % 1) * 1000)
    total_sec = int(seconds)
    return f"{total_sec // 3600:02d}:{(total_sec % 3600) // 60:02d}:{total_sec % 60:02d},{millis:03d}"


async def generate_tts(text: str, out_path: Path, voice: str) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))


def generate_image(prompt: str, out_path: Path) -> None:
    result = ai_client.models.generate_images(
        model="imagen-3.0-generate-002",
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            output_mime_type="image/jpeg",
            aspect_ratio="9:16",
            person_generation="ALLOW_ADULT",
        )
    )
    for img in result.generated_images:
        with open(out_path, "wb") as f:
            f.write(img.image.image_bytes)
        break


def render_clip(img_path: Path, aud_path: Path, out_clip: Path, duration: float) -> None:
    fps = 30
    frames = int(duration * fps) + 5
    vf = (
        f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
        f"zoompan=z='min(zoom+0.0005,1.08)':d={frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920:fps={fps}"
    )
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", str(img_path), "-i", str(aud_path),
        "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-t", f"{duration:.3f}", str(out_clip)
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


async def process_video_job(job_id: str, title: str, scenes: List[SceneInput], voice: str) -> None:
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        JOBS[job_id]["status"] = "processing"
        clips = []
        srt_lines = []
        curr_time = 0.0

        for idx, scene in enumerate(scenes, start=1):
            JOBS[job_id]["progress"] = f"กำลังสร้างฉากที่ {idx}/{len(scenes)}..."
            aud_file = job_dir / f"s_{idx}.mp3"
            img_file = job_dir / f"s_{idx}.jpg"
            clip_file = job_dir / f"s_{idx}.mp4"

            await generate_tts(scene.narration, aud_file, voice)
            dur = get_audio_duration(aud_file)
            await asyncio.to_thread(generate_image, scene.visual_prompt, img_file)
            await asyncio.to_thread(render_clip, img_file, aud_file, clip_file, dur)

            clips.append(clip_file)
            srt_lines.append(f"{idx}\n{format_srt_time(curr_time)} --> {format_srt_time(curr_time + dur)}\n{scene.narration}\n")
            curr_time += dur

        JOBS[job_id]["progress"] = "กำลังรวมไฟล์และฝังซับไตเติล..."
        concat_txt = job_dir / "concat.txt"
        with open(concat_txt, "w", encoding="utf-8") as f:
            for c in clips:
                f.write(f"file '{str(c).replace(chr(92), '/')}'\n")

        merged = job_dir / "merged.mp4"
        await asyncio.to_thread(
            subprocess.run,
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", str(merged)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )

        srt_file = job_dir / "sub.srt"
        with open(srt_file, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_lines))

        filename = f"video_{job_id[:8]}.mp4"
        final_mp4 = OUTPUT_DIR / filename
        esc_srt = str(srt_file).replace("\\", "/").replace(":", "\\:")
        style = "FontName=Waree,FontSize=15,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Alignment=2,MarginV=65"

        await asyncio.to_thread(
            subprocess.run,
            ["ffmpeg", "-y", "-i", str(merged), "-vf", f"subtitles='{esc_srt}':force_style='{style}'", "-c:v", "libx264", "-preset", "fast", "-c:a", "copy", str(final_mp4)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["progress"] = "เสร็จสมบูรณ์!"
        JOBS[job_id]["filename"] = filename

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)


@mcp.tool()
async def create_video(title: str, scenes: List[SceneInput], voice_gender: str = "male") -> str:
    """
    สั่งสร้างวิดีโอสั้นแนวตั้ง 9:16 พร้อมเสียงพากย์ไทยและภาพประกอบจาก Gemini
    คืนค่าเป็นลิงก์หน้าเว็บสำหรับติดตามสถานะและกดดาวน์โหลดวิดีโอบน iPad
    """
    job_id = str(uuid.uuid4())
    voice = "th-TH-NiwatNeural" if voice_gender.lower() == "male" else "th-TH-PremwadeeNeural"

    JOBS[job_id] = {
        "title": title,
        "status": "processing",
        "progress": "กำลังเริ่มต้นสร้างวิดีโอ...",
        "filename": None,
        "error": None
    }

    asyncio.create_task(process_video_job(job_id, title, scenes, voice))
    watch_url = f"{PUBLIC_URL}/watch/{job_id}"

    return (
        f"เริ่มสร้างวิดีโอ '{title}' แล้ว!\n"
        f"กดเปิดลิงก์นี้เพื่อดูความคืบหน้าและดาวน์โหลดวิดีโอได้เลย:\n{watch_url}\n"
        f"(หรือจะให้ฉันเช็กสถานะให้ผ่าน check_status ด้วยรหัส {job_id} ก็ได้)"
    )


@mcp.tool()
def check_status(job_id: str) -> str:
    """เช็กสถานะการสร้างวิดีโอและรับลิงก์หน้าเว็บ"""
    job = JOBS.get(job_id)
    if not job:
        return "ไม่พบรหัสงานนี้"
    watch_url = f"{PUBLIC_URL}/watch/{job_id}"
    if job["status"] == "completed":
        return f"วิดีโอสร้างเสร็จแล้ว! กดดูและดาวน์โหลดได้ที่ลิงก์นี้:\n{watch_url}"
    elif job["status"] == "failed":
        return f"เกิดข้อผิดพลาด: {job['error']}"
    return f"กำลังทำอยู่ ({job['progress']}) สามารถเปิดรอที่หน้าเว็บนี้ได้เลย: {watch_url}"


@app.get("/watch/{job_id}", response_class=HTMLResponse)
async def watch_page(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return "<h1>ไม่พบวิดีโอนี้</h1>"

    if job["status"] == "completed":
        video_url = f"/files/{job['filename']}"
        return f"""
        <!DOCTYPE html>
        <html lang="th">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{job['title']}</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-slate-950 text-white min-h-screen flex flex-col items-center justify-center p-4">
            <div class="max-w-sm w-full bg-slate-900 rounded-2xl p-5 shadow-xl border border-slate-800 text-center">
                <h1 class="text-lg font-bold mb-3 text-emerald-400">สร้างวิดีโอสำเร็จ!</h1>
                <p class="text-sm text-slate-300 mb-4">{job['title']}</p>
                <video src="{video_url}" controls playsinline class="w-full rounded-xl mb-5 bg-black aspect-[9/16]"></video>
                <a href="{video_url}" download="{job['filename']}" 
                   class="block w-full bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-bold py-3 px-4 rounded-xl transition">
                   บันทึกลง iPad (Download MP4)
                </a>
            </div>
        </body>
        </html>
        """
    elif job["status"] == "failed":
        return f"<body style='background:#0f172a;color:#f87171;padding:2rem;font-family:sans-serif;'><h2>เกิดข้อผิดพลาด</h2><p>{job['error']}</p></body>"
    else:
        return f"""
        <!DOCTYPE html>
        <html lang="th">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <meta http-equiv="refresh" content="4">
            <title>กำลังสร้างวิดีโอ...</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-slate-950 text-white min-h-screen flex flex-col items-center justify-center p-4">
            <div class="max-w-sm w-full bg-slate-900 rounded-2xl p-6 text-center border border-slate-800">
                <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-400 mx-auto mb-4"></div>
                <h2 class="text-lg font-bold mb-2">กำลังเรนเดอร์วิดีโอของคุณ</h2>
                <p class="text-sm text-slate-400">{job['progress']}</p>
                <p class="text-xs text-slate-500 mt-4">หน้านี้จะอัปเดตอัตโนมัติทุก 4 วินาที...</p>
            </div>
        </body>
        </html>
        """

mcp_asgi = mcp.streamable_http_app()
app.add_route("/mcp", mcp_asgi, methods=["GET", "POST", "DELETE", "OPTIONS"])
app.mount("/mcp", mcp_asgi)

