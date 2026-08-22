import os
import pandas as pd
import gradio as gr
import re
import google.generativeai as genai
from PIL import Image
import io
import numpy as np
from gtts import gTTS
from sklearn.cluster import KMeans
import random

# 1. Setup Google Gemini with Load Balancing (3 API Keys)
# This rotates keys so 10 users can use the app simultaneously without hitting limits
API_KEYS = [
    "AQ.Ab8RN6KiB9HBDhhRmWPWizkb_Z9pBaFkt3BlK4JhmIKZofiMHA",
    "AQ.Ab8RN6JaeGTdVaZsBn7Xf8AfE6KAmrSgoR-cZnGpX5za63T43Q",
    "AQ.Ab8RN6KHWcBqABaBFh0QGE9by37e9a16S1vJhglTVAyy9Trt_A"
]

# Pick a random key for this session to distribute the load
selected_key = random.choice(API_KEYS)
genai.configure(api_key=selected_key)
model = genai.GenerativeModel('gemini-2.0-flash') # 2.0-flash has the best free-tier limits

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

def get_color_percentages(img_path):
    """Calculates exact percentages in 0.05 seconds on a 30x30 image"""
    try:
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((30, 30)) 
        arr = np.array(img).reshape((-1, 3))
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=1, max_iter=10).fit(arr)
        counts = np.unique(kmeans.labels_, return_counts=True)[1]
        pcts = (counts / counts.sum()) * 100
        colors = kmeans.cluster_centers_.astype(int)
        idx = np.argsort(-pcts); colors = colors[idx]; pcts = pcts[idx]
        
        data = []
        for i in range(len(colors)):
            r, g, b = colors[i]
            data.append(f"Color {i+1}: RGB({r},{g},{b}) - {pcts[i]:.1f}%")
        return "\n".join(data)
    except:
        return "Color data unavailable."

def compress_image_for_gemini(img_path):
    """Compresses image to 512px to prevent RAM crashes and speed up API"""
    try:
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=65)
        return buffer.getvalue()
    except:
        with open(img_path, "rb") as f: return f.read()

def answer_question(message, history):
    if df.empty: return "Error loading data."
    
    user_text = message.get("text", "").strip()
    nums = re.findall(r'\b(\d+)\b', user_text)
    art_id = None
    for n in nums:
        if 1 <= int(n) <= 53:
            art_id = int(n)
            break

    # SCENARIO A: No number provided (Handles BOTH Follow-ups AND General Questions)
    if not art_id:
        if not user_text: return "Please enter a number 1-53."
        
        csv_data = df.to_string(index=False)
        
        is_followup = False
        if history:
            matches = re.findall(r'Artwork ID (\d+)', str(history))
            if matches:
                art_id = int(matches[-1])
                is_followup = True

        if is_followup:
            row = df[df['ID'].astype(str).str.strip() == str(art_id)].iloc[0]
            title = str(row.get('TITLE', 'Unknown'))
            orig_path = get_image_path(art_id)
            img_bytes = compress_image_for_gemini(orig_path) if orig_path else None

            prompt = f"""
            The user asked: "{user_text}"
            You are currently analyzing Artwork ID {art_id} ({title}). The image is attached.
            The full database metadata for all 53 artworks is provided below:
            {csv_data}

            INSTRUCTIONS:
            - If the user is asking a follow-up question or challenging your previous analysis about Artwork {art_id}, respond conversationally in under 150 words based on the attached image.
            - If the user is asking a GENERAL question (e.g., "which ones have boats or animals?"), IGNORE the attached image and answer their question by searching the database metadata provided above. List the Artwork IDs and Titles.
            - YOU MUST NOT HALLUCINATE. Use only the provided data.
            """
            try:
                if img_bytes:
                    res = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": img_bytes}])
                else:
                    res = model.generate_content(prompt)
                res_text = res.text

                audio_path = "followup_response.mp3"
                clean_text = re.sub(r'[*#`_]', '', res_text)
                gTTS(clean_text, lang='en', slow=False).save(audio_path)

                return {"text": res_text, "files": [audio_path]}
            except Exception as e:
                return f"Error: {str(e)}"
        else:
            prompt = f"""
            The user asked: "{user_text}"
            Here is the archival database metadata for all 53 artworks:
            {csv_data}
            Instructions: Answer the user's question using ONLY the database metadata provided above. List Artwork IDs and Titles. DO NOT HALLUCINATE.
            """
            res = model.generate_content(prompt)
            res_text = res.text
            audio_path = "general_response.mp3"
            clean_text = re.sub(r'[*#`_]', '', res_text)
            gTTS(clean_text, lang='en', slow=False).save(audio_path)
            return {"text": res_text, "files": [audio_path]}

    # SCENARIO B: Fresh request for a Single Artwork (Strict 600+ word analysis)
    row = df[df['ID'].astype(str).str.strip() == str(art_id)].iloc[0]
    title = str(row.get('TITLE', 'Unknown'))
    
    # EXTRACT METADATA TO PREVENT HALLUCINATIONS
    artist = str(row.get('Artist (if known)', 'Unknown'))
    date = str(row.get('Date', 'Unknown'))
    style = str(row.get('Artistic style', 'Unknown'))
    source = str(row.get('Source', 'N/A'))
    csv_context = f"Title: {title}\nArtist: {artist}\nDate: {date}\nStyle: {style}\nSource: {source}"
    
    orig_path = get_image_path(art_id)
    seg_path = f"preloaded_segments/seg_{art_id}.jpg"
    col_path = f"preloaded_colors/colors_{art_id}.jpg"
    audio_path = f"art_{art_id}_audio.mp3"

    # CALCULATE EXACT COLOR PERCENTAGES
    color_data = get_color_percentages(orig_path)

    # Compress image for Gemini
    gemini_img_bytes = compress_image_for_gemini(orig_path) if orig_path else None

    prompt = f"""
    You are a strict, analytical art historian. You are analyzing Artwork ID {art_id}.
    
    Here is the EXACT archival data for this artwork:
    {csv_context}

    Here is the quantitative data for the 5 most dominant colors extracted via K-Means clustering:
    {color_data}

    CRITICAL RULES (STRICTLY ENFORCED):
    1. YOU MUST NOT HALLUCINATE. Do not use any outside knowledge. If the archival data says 'Unknown' for the Artist, you MUST state "Artist Unknown". Do not invent names, dates, or historical facts not present in the archival data.
    2. YOUR RESPONSE MUST BE EXACTLY FOUR PARAGRAPHS. EACH PARAGRAPH MUST BE AT LEAST 150 WORDS. THE TOTAL RESPONSE MUST BE OVER 600 WORDS.
    3. Do not provide generic introductions or conclusions. Go directly into deep, analytical prose.

    PARAGRAPH STRUCTURE AND REQUIREMENTS:
    - Paragraph 1 (Archival & Contextual Analysis): Analyze the artwork using ONLY the archival data provided above. Discuss the title, artist (if known), date, artistic style, and source. Do not invent historical context; rely strictly on the provided metadata.
    - Paragraph 2 (Quantitative Visual & Color Analysis): Analyze the visual composition of the attached image. You MUST explicitly state and analyze the exact RGB values and Percentages provided in the quantitative color data above. Discuss what these dominant colors signify regarding mood, lighting, and materiality based strictly on what you see in the attached image.
    - Paragraph 3 (Urban & Environmental Context): Relate the artwork to the urban and environmental history of Adelaide based strictly on the visual evidence (e.g., infrastructure, landscape, River Torrens, colonial settlement) and the archival date. Do not invent historical figures.
    - Paragraph 4 (Semantic Segmentation Analysis): You are looking at the original image. Conduct a rigorous textual analysis of how a semantic segmentation algorithm would break down this image. Discuss the distinct spatial regions, boundaries, and color fields (e.g., sky, water, land, architecture, figures). Explain what this computational breakdown reveals about the composition and spatial hierarchy of the artwork.
    """
    
    try:
        res = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": gemini_img_bytes}])
        res_text = res.text

        # Generate Audio File
        clean_text = re.sub(r'[*#`_]', '', res_text)
        gTTS(clean_text, lang='en', slow=False).save(audio_path)

        # Build the text response
        text_md = f"**Artwork ID {art_id}**\n\n{res_text}\n\n---\n"
        text_md += f"**Quantitative Color Data:**\n{color_data}\n\n---\n"
        
        # Collect all file paths to send back to Gradio natively (3 images + 1 audio)
        files_to_return = []
        if orig_path and os.path.exists(orig_path): files_to_return.append(orig_path)
        if os.path.exists(seg_path): files_to_return.append(seg_path)
        if os.path.exists(col_path): files_to_return.append(col_path)
        files_to_return.append(audio_path)

        return {
            "text": text_md,
            "files": files_to_return
        }
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Gradio Interface
demo = gr.ChatInterface(fn=answer_question, multimodal=True, title="Adelaide Artworks AI (1-53)")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
