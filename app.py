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
    print(f"Error loading Excel: {e}")
    df = pd.DataFrame()

def get_image_data(image_id):
    img_dir = "."
    for filename in os.listdir(img_dir):
        ext = filename.split('.')[-1].lower()
        if filename.lower().startswith(f"{image_id}.") and ext in ["jpg", "jpeg", "png", "webp"]:
            img_path = os.path.join(img_dir, filename)
            with open(img_path, "rb") as image_file: return img_path, image_file.read()
    return None, None

def optimize_image_for_gemini(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        byte_arr = io.BytesIO()
        img.save(byte_arr, format='JPEG', quality=70)
        return byte_arr.getvalue()
    except: return img_bytes

def text_to_speech_html(text, filename="audio.mp3"):
    try:
        clean_text = re.sub(r'[*#`_]', '', text)
        tts = gTTS(clean_text, lang='en', slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio_b64 = base64.b64encode(mp3_fp.read()).decode()
        return f"""
        <div style="margin-top: 15px;">
            <a href="data:audio/mp3;base64,{audio_b64}" download="{filename}" style="padding: 10px 15px; background-color: #2563eb; color: white; text-decoration: none; border-radius: 5px; display: inline-block; margin-bottom: 10px; font-weight: bold;">⬇ Download Audio</a>
            <audio controls autoplay style="width: 100%; display: block;"><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>
        </div>
        """
    except: return ""

def pil_to_base64_html(img, filename, caption):
    if img is None: return ""
    try:
        buffered = io.BytesIO()
        img.convert("RGB").save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"""
        <div style="margin-top: 15px;">
            <p style="font-weight: bold; margin-bottom: 5px;">{caption} (Click to download)</p>
            <a href="data:image/jpeg;base64,{img_str}" download="{filename}">
                <img src="data:image/jpeg;base64,{img_str}" alt="{caption}" style="width: 100%; border-radius: 8px; border: 1px solid #444; cursor: pointer;">
            </a>
        </div>
        """
    except: return ""

# 3. The Chat Engine
def answer_question(message, history):
    if df.empty: return {"text": "Error: Could not load data.xlsx."}

    user_text = message.get("text", "").strip()
    requested_ids = []
    for num in re.findall(r'\b(\d+)\b', user_text):
        if 1 <= int(num) <= 53: requested_ids.append(int(num))
    requested_ids = list(dict.fromkeys(requested_ids))

    if not requested_ids and history:
        matches = re.findall(r'Artwork ID (\d+)', str(history))
        if matches: requested_ids = [int(matches[-1])]

    if not requested_ids:
        csv_data = df.to_string(index=False)
        prompt = f"The user asked: '{user_text}'\nHere is the archival database for 53 artworks:\n{csv_data}\nInstructions: Answer based on the database. Provide IDs and Titles."
        try:
            res_text = model.generate_content(prompt).text
            return {"text": res_text + text_to_speech_html(res_text, "answer.mp3")}
        except Exception as e: return {"text": f"Error: {str(e)}"}

    # Load Pre-processed Images (Instant, 0 seconds)
    requested_id = requested_ids[0]
    match_df = df[df['ID'].astype(str).str.strip() == str(requested_id)]
    if match_df.empty: return {"text": "Artwork not found."}

    row = match_df.iloc[0]
    title = str(row.get('TITLE', 'Unknown'))
    img_path, img_bytes = get_image_data(requested_id)
    
    # Load pre-made segmentation and color bar
    seg_path = f"preloaded_segments/seg_{requested_id}.jpg"
    colors_path = f"preloaded_colors/colors_{requested_id}.jpg"
    
    seg_img = Image.open(seg_path) if os.path.exists(seg_path) else None
    color_img = Image.open(colors_path) if os.path.exists(colors_path) else None

    prompt = f"""
    You are an expert art historian. Artwork ID {requested_id}: {title}, Artist: {row.get('Artist (if known)', 'Unknown')}, Date: {row.get('Date', 'Unknown')}.
    RULES:
    1. EXACTLY THREE PARAGRAPHS, UNDER 300 WORDS TOTAL.
    2. Paragraph 1: Introduce artwork and visual analysis.
    3. Paragraph 2: Relate to urban history of Adelaide.
    4. Paragraph 3: Analyze semantic segmentation and dominant colors.
    """
    try:
        # ONLY Gemini is called here. No heavy math. Takes ~15 seconds.
        response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": optimize_image_for_gemini(img_bytes)}])
        res_text = response.text
        
        text_md = f"**Artwork ID {requested_id}**\n\n{res_text}\n\n---\n"
        text_md += pil_to_base64_html(Image.open(io.BytesIO(img_bytes)), f"art_{requested_id}_orig.jpg", "Original Artwork")
        text_md += pil_to_base64_html(seg_img, f"art_{requested_id}_seg.jpg", "Semantic Segmentation")
        text_md += pil_to_base64_html(color_img, f"art_{requested_id}_colors.jpg", "Dominant Color Palette")

        return {"text": text_md + text_to_speech_html(res_text, f"art_{requested_id}.mp3")}
    except Exception as e:
        return {"text": f"Error: {str(e)}"}

# 4. Gradio Interface
demo = gr.ChatInterface(
    fn=answer_question,
    multimodal=True,
    title="Adelaide Artworks AI (1-53)",
    description="Ask about an artwork (1-53)!"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
