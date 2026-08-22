import os
import pandas as pd
import gradio as gr
import re
import google.generativeai as genai
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

    # SCENARIO A: General Question (No number)
    if not art_id:
        if not user_text: return "Please enter a number 1-53."
        res = model.generate_content(f"User asked: '{user_text}'\nDatabase:\n{df.to_string(index=False)}\nAnswer:")
        
        # Save audio to file so Gradio can render it
        audio_path = "general_response.mp3"
        clean_text = re.sub(r'[*#`_]', '', res.text)
        gTTS(clean_text, lang='en', slow=False).save(audio_path)
        
        return {"text": res.text, "files": [audio_path]}

    # SCENARIO B: Specific Artwork Request
    row = df[df['ID'].astype(str).str.strip() == str(art_id)].iloc[0]
    title = str(row.get('TITLE', 'Unknown'))
    
    orig_path = get_image_path(art_id)
    seg_path = f"preloaded_segments/seg_{art_id}.jpg"
    col_path = f"preloaded_colors/colors_{art_id}.jpg"
    audio_path = f"art_{art_id}_audio.mp3"

    # Send the tiny pre-processed seg image to Gemini to prevent RAM crashes
    gemini_img_bytes = None
    if os.path.exists(seg_path):
        with open(seg_path, "rb") as f:
            gemini_img_bytes = f.read()
    elif orig_path and os.path.exists(orig_path):
        with open(orig_path, "rb") as f:
            gemini_img_bytes = f.read()

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

    # Generate Audio File
    clean_text = re.sub(r'[*#`_]', '', res_text)
    gTTS(clean_text, lang='en', slow=False).save(audio_path)

    # Build the text response
    text_md = f"**Artwork ID {art_id}**\n\n{res_text}\n\n---\n"
    
    # Collect all file paths to send back to Gradio natively
    files_to_return = []
    if orig_path and os.path.exists(orig_path): files_to_return.append(orig_path)
    if os.path.exists(seg_path): files_to_return.append(seg_path)
    if os.path.exists(col_path): files_to_return.append(col_path)
    files_to_return.append(audio_path)

    # Return the dictionary using the "files" key, which Gradio natively renders!
    return {
        "text": text_md,
        "files": files_to_return
    }

# Gradio Interface
demo = gr.ChatInterface(fn=answer_question, multimodal=True, title="Adelaide Artworks AI (1-53)")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
