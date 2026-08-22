import os
import pandas as pd
import gradio as gr
import re
import base64
import google.generativeai as genai
from PIL import Image
import io
from gtts import gTTS

# 1. Setup Google Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-3.7-flash')

# 2. Load Excel Data
try:
    df = pd.read_excel("data.xlsx")
    df.columns = df.columns.str.strip()
except Exception as e:
    df = pd.DataFrame()

def get_image_path(image_id):
    for filename in os.listdir("."):
        if filename.lower().startswith(f"{image_id}.") and filename.split('.')[-1].lower() in ["jpg", "jpeg", "png", "webp"]:
            return filename
    return None

def make_html_image(filepath, caption):
    if not filepath or not os.path.exists(filepath): return ""
    try:
        img = Image.open(filepath).convert("RGB")
        img.thumbnail((500, 500), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=80)
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"""
        <div style="margin-top:15px;">
            <p style="font-weight:bold;">{caption} (Click to download)</p>
            <a href="data:image/jpeg;base64,{b64}" download="{os.path.basename(filepath)}">
                <img src="data:image/jpeg;base64,{b64}" style="width:100%; border-radius:8px; border:1px solid #444;">
            </a>
        </div>
        """
    except:
        return ""

def make_audio_html(text, filename):
    try:
        clean = re.sub(r'[*#`_]', '', text)
        tts = gTTS(clean, lang='en', slow=False)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()
        return f"""
        <div style="margin-top:15px;">
            <a href="data:audio/mp3;base64,{b64}" download="{filename}" style="padding:10px; background:#2563eb; color:white; text-decoration:none; border-radius:5px; font-weight:bold;">⬇ Download Audio</a>
            <audio controls autoplay style="width:100%; margin-top:10px;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>
        </div>
        """
    except:
        return ""

def answer_question(message, history):
    if df.empty: return "Error loading data."
    
    user_text = message.get("text", "").strip()
    nums = re.findall(r'\b(\d+)\b', user_text)
    art_id = None
    for n in nums:
        if 1 <= int(n) <= 53:
            art_id = int(n)
            break
            
    if not art_id and history:
        matches = re.findall(r'Artwork ID (\d+)', str(history))
        if matches: art_id = int(matches[-1])

    if not art_id:
        if not user_text: return "Please enter a number 1-53."
        res = model.generate_content(f"User asked: '{user_text}'\nDatabase:\n{df.to_string(index=False)}\nAnswer:")
        return res.text + make_audio_html(res.text, "response.mp3")

    row = df[df['ID'].astype(str).str.strip() == str(art_id)].iloc[0]
    title = str(row.get('TITLE', 'Unknown'))
    
    orig_path = get_image_path(art_id)
    seg_path = f"preloaded_segments/seg_{art_id}.jpg"
    col_path = f"preloaded_colors/colors_{art_id}.jpg"

    # Shrink image for Gemini
    try:
        img = Image.open(orig_path).convert("RGB")
        img.thumbnail((256, 256), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=60)
        gemini_img_bytes = buffer.getvalue()
    except:
        gemini_img_bytes = None

    prompt = f"""
    You are an expert art historian. Artwork ID {art_id}: {title}. 
    Provide EXACTLY FOUR paragraphs (about 80 words each).
    1. Introduce the artwork (Name, Artist, Year, context) and visual analysis.
    2. Conduct a visual analysis of the attached image, including dominant colors and mood.
    3. Relate to urban history of Adelaide.
    4. Conduct a textual analysis of semantic segmentation (sky, water, land, etc.).
    """
    
    res = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": gemini_img_bytes}])
    res_text = res.text

    html = f"**Artwork ID {art_id}**\n\n{res_text}\n\n---\n"
    html += make_html_image(orig_path, "Original Artwork")
    html += make_html_image(seg_path, "Semantic Segmentation")
    html += make_html_image(col_path, "Dominant Color Palette")
    html += make_audio_html(res_text, f"art_{art_id}.mp3")

    return html

# Added type="messages" back so Gradio knows how to read the multimodal input!
demo = gr.ChatInterface(fn=answer_question, type="messages", multimodal=True, title="Adelaide Artworks AI (1-53)")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
