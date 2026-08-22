import os
import pandas as pd
import gradio as gr
import re
import base64
import google.generativeai as genai
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

def get_raw_bytes(filepath):
    """Reads a file instantly as raw bytes without PIL processing"""
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return f.read()
    return None

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
    except:
        return ""

def bytes_to_base64_html(img_bytes, filename, caption):
    """Instantly converts raw bytes to HTML for display and download"""
    if img_bytes is None: return ""
    try:
        img_str = base64.b64encode(img_bytes).decode('utf-8')
        return f"""
        <div style="margin-top: 15px;">
            <p style="font-weight: bold; margin-bottom: 5px;">{caption} (Click to download)</p>
            <a href="data:image/jpeg;base64,{img_str}" download="{filename}">
                <img src="data:image/jpeg;base64,{img_str}" alt="{caption}" style="width: 100%; border-radius: 8px; border: 1px solid #444; cursor: pointer;">
            </a>
        </div>
        """
    except:
        return ""

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

    requested_id = requested_ids[0]
    match_df = df[df['ID'].astype(str).str.strip() == str(requested_id)]
    if match_df.empty: return {"text": "Artwork not found."}

    row = match_df.iloc[0]
    title = str(row.get('TITLE', 'Unknown'))
    
    # Find the original image extension
    img_dir = "."
    original_img_path = None
    for filename in os.listdir(img_dir):
        if filename.lower().startswith(f"{requested_id}.") and filename.split('.')[-1].lower() in ["jpg", "jpeg", "png", "webp"]:
            original_img_path = os.path.join(img_dir, filename)
            break

    if not original_img_path:
        return {"text": "Image not found."}

    # INSTANTLY load pre-made images from folders (Zero CPU math)
    original_bytes = get_raw_bytes(original_img_path)
    seg_bytes = get_raw_bytes(f"preloaded_segments/seg_{requested_id}.jpg")
    colors_bytes = get_raw_bytes(f"preloaded_colors/colors_{requested_id}.jpg")
    
    # Send the SMALL seg_bytes to Gemini to analyze. 
    # This removes the need for PIL completely! Zero resizing on Render's CPU.
    image_to_send_to_gemini = seg_bytes if seg_bytes else original_bytes

    prompt = f"""
    You are an expert art historian. Artwork ID {requested_id}: {title}, Artist: {row.get('Artist (if known)', 'Unknown')}, Date: {row.get('Date', 'Unknown')}.
    RULES:
    1. YOUR RESPONSE MUST BE EXACTLY FOUR PARAGRAPHS.
    2. EACH PARAGRAPH MUST BE APPROXIMATELY 100 WORDS. (Total 400 words).
    3. Paragraph 1: Introduce the artwork (Name, Artist, Year, context) and visual analysis.
    4. Paragraph 2: Conduct a visual analysis of the attached image, including dominant colors and mood.
    5. Paragraph 3: Relate to urban history of Adelaide.
    6. Paragraph 4: Conduct a textual analysis of semantic segmentation (sky, water, land, etc.).
    """
    try:
        # Only Gemini API call + Audio generation. ZERO Image processing.
        response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": image_to_send_to_gemini}])
        res_text = response.text
        
        text_md = f"**Artwork ID {requested_id}**\n\n{res_text}\n\n---\n"
        text_md += bytes_to_base64_html(original_bytes, f"art_{requested_id}_orig.jpg", "Original Artwork")
        text_md += bytes_to_base64_html(seg_bytes, f"art_{requested_id}_seg.jpg", "Semantic Segmentation")
        text_md += bytes_to_base64_html(colors_bytes, f"art_{requested_id}_colors.jpg", "Dominant Color Palette")

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
