import os
import pandas as pd
import gradio as gr
import re
import base64
import google.generativeai as genai
from PIL import Image
import io
from skimage import segmentation, color
import numpy as np
from gtts import gTTS # New: For Text-to-Speech

# 1. Setup Google Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

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
            with open(img_path, "rb") as image_file:
                img_bytes = image_file.read()
            mime_type = {
                "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "webp": "image/webp"
            }.get(ext, "image/jpeg")
            return img_path, img_bytes, mime_type
    return None, None, "Error: Image file not found."

def generate_segmentation_image(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.thumbnail((800, 800), Image.Resampling.LANCZOS)
        img_array = np.array(img)
        segments = segmentation.slic(img_array, n_segments=100, compactness=10, start_label=1)
        segmented_img = color.label2rgb(segments, img_array, kind='avg', bg_label=0)
        seg_pil = Image.fromarray((segmented_img * 255).astype(np.uint8))
        byte_arr = io.BytesIO()
        seg_pil.save(byte_arr, format='JPEG', quality=85)
        return byte_arr.getvalue()
    except Exception as e:
        print(f"Segmentation error: {e}")
        return None

def text_to_speech_html(text):
    """Converts text to speech and returns an HTML audio tag."""
    try:
        tts = gTTS(text, lang='en', slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio_b64 = base64.b64encode(mp3_fp.read()).decode()
        audio_html = f"\n\n<audio controls autoplay style='width:100%;'><source src='data:audio/mp3;base64,{audio_b64}' type='audio/mp3'></audio>"
        return audio_html
    except Exception as e:
        print(f"TTS Error: {e}")
        return ""

# 3. The Multimodal Chat Engine
def answer_question(message, history):
    if df.empty:
        return {"text": "Error: Could not load data.xlsx."}

    # Extract text and uploaded files from Gradio multimodal input
    user_text = message.get("text", "")
    user_files = message.get("files", [])

    # Find Artwork ID in current message
    requested_id = None
    numbers = re.findall(r'\b(\d+)\b', user_text)
    for num in numbers:
        if 1 <= int(num) <= 53: # Updated to 53
            requested_id = int(num)
            break

    # If no ID in current message, check history for context
    is_followup = False
    if not requested_id and history:
        # Scan history string to find the last mentioned Artwork ID
        history_str = str(history)
        matches = re.findall(r'Artwork ID (\d+)', history_str)
        if matches:
            requested_id = int(matches[-1])
            is_followup = True

    # SCENARIO A: User uploaded an image/file
    if user_files:
        parts = [f"User said: '{user_text}'. The user also uploaded the following image(s) for you to analyze."]
        
        # If we are currently discussing an artwork, include it for comparison
        if requested_id:
            img_path, img_bytes, mime_type = get_image_data(requested_id)
            if img_bytes:
                parts.append(f"Reference Artwork ID {requested_id}:")
                parts.append({"mime_type": mime_type, "data": img_bytes})

        # Attach user uploaded files
        for file_path in user_files:
            ext = file_path.split('.')[-1].lower()
            mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
            with open(file_path, "rb") as f:
                parts.append({"mime_type": mime, "data": f.read()})
                parts.append("User uploaded image.")

        try:
            response = model.generate_content(parts)
            res_text = response.text
            audio_html = text_to_speech_html(res_text)
            return {"text": res_text + audio_html}
        except Exception as e:
            return {"text": f"Error analyzing uploaded image: {str(e)}"}

    # SCENARIO B: No artwork ID found at all
    if not requested_id:
        return {"text": "Please enter a valid artwork number between **1 and 53**, or upload an image to discuss."}

    # Load Artwork Data
    match_df = df[df['ID'].astype(str).str.strip() == str(requested_id)]
    if match_df.empty:
        return {"text": f"Could not find data for Artwork ID {requested_id}."}

    row = match_df.iloc[0]
    title = str(row.get('TITLE', 'Unknown Title'))
    date = str(row.get('Date', 'Unknown Date'))
    artist = str(row.get('Artist (if known)', 'Unknown Artist'))
    style = str(row.get('Artistic style', 'Unknown Style'))

    img_path, img_bytes, mime_or_error = get_image_data(requested_id)
    if not img_bytes:
        return {"text": f"**Artwork ID {requested_id}:** {title} ({date}) by {artist}.\n\n*({mime_or_error})*"}

    csv_context = f"Title: {title}\nArtist: {artist}\nDate: {date}\nStyle: {style}\nSource: {row.get('Source', 'N/A')}"

    # SCENARIO C: Follow-up conversational question
    if is_followup:
        prompt = f"""
        The user is asking a follow-up question about Artwork ID {requested_id} ({title}).
        Archival Data: {csv_context}
        User's new input: "{user_text}"
        
        Instructions: Respond conversationally to the user's input. Do not repeat the 4 paragraphs. Address their specific agreement, disagreement, or question directly. 
        """
        try:
            response = model.generate_content([
                prompt,
                {"mime_type": mime_or_error, "data": img_bytes}
            ])
            res_text = response.text
            audio_html = text_to_speech_html(res_text)
            return {"text": res_text + audio_html}
        except Exception as e:
            return {"text": f"Error: {str(e)}"}

    # SCENARIO D: Fresh request for an Artwork (Standard 4-paragraph response)
    strict_prompt = f"""
    You are an expert art historian. The user requested information about Artwork ID {requested_id}.
    Archival data:
    {csv_context}

    RULES:
    1. YOUR RESPONSE MUST BE EXACTLY FOUR PARAGRAPHS.
    2. Paragraph 1: Introduce the artwork (Name, Artist, Year, context).
    3. Paragraph 2: Visual analysis of the attached image.
    4. Paragraph 3: Relate to urban history of Adelaide.
    5. Paragraph 4: Textual analysis of semantic segmentation (sky, water, land, etc.).
    """
    try:
        response = model.generate_content([
            strict_prompt,
            {"mime_type": mime_or_error, "data": img_bytes}
        ])
        response_text = response.text
        seg_bytes = generate_segmentation_image(img_bytes)

        # Build the text response
        text_md = f"**Artwork ID {requested_id}**\n\n{response_text}\n\n---\n"
        text_md += "**Original Artwork & Semantic Segmentation Map:**\n"

        # Generate Audio
        audio_html = text_to_speech_html(response_text)

        # Combine text, images, and audio
        if seg_bytes:
            return {
                "text": text_md + audio_html,
                "images": [img_bytes, seg_bytes]
            }
        else:
            return {
                "text": text_md + "\n*(Segmentation image could not be generated)*" + audio_html,
                "images": [img_bytes]
            }
    except Exception as e:
        return {"text": f"Error generating response: {str(e)}"}

# 4. Gradio Interface
demo = gr.ChatInterface(
    fn=answer_question,
    type="messages",
    multimodal=True, # ✅ Enables the file upload button!
    title="Adelaide Artworks AI (1-53)",
    description="Enter a number (1-53) to view artwork & analysis. You can also upload your own images to discuss, or ask follow-up questions!",
    textbox=gr.Textbox(placeholder="Enter a number 1-53, or ask a follow-up question..."),
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
