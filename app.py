import os
import pandas as pd
import gradio as gr
import re
import traceback
from google import genai
from google.genai import types
from PIL import Image
import io

# 1. Setup Google Gemini 
client = genai.Client(api_key=os.environ.get("gemini_API_KEY"))

# 2. Load Excel Data
try:
    df = pd.read_excel("data.xlsx")
    df.columns = df.columns.str.strip()
except Exception as e:
    print("Error loading data.xlsx:", e)
    df = pd.DataFrame()

# 3. Pre-compute Image Lookup Dictionary (Performance Fix)
IMAGE_LOOKUP = {}
for filename in os.listdir("."):
    if filename.split('.')[-1].lower() in ["jpg", "jpeg", "png", "webp"]:
        base_name = filename.split('.')[0]
        IMAGE_LOOKUP[base_name] = filename

def get_image_path(image_id):
    return IMAGE_LOOKUP.get(str(image_id))

def compress_image_for_gemini(img_path):
    """Compresses image to 512px to prevent RAM crashes and speed up API"""
    try:
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=65)
        return buffer.getvalue()
    except Exception:
        with open(img_path, "rb") as f: 
            return f.read()

def answer_question(user_text, history):
    if df.empty: 
        return "Error loading data."
    
    user_text = user_text.strip()
    nums = re.findall(r'\b(\d+)\b', user_text)
    art_id = None
    for n in nums:
        if 1 <= int(n) <= 53:
            art_id = int(n)
            break

    # SCENARIO A: No number provided (Handles BOTH Follow-ups AND General Questions)
    if not art_id:
        if not user_text: 
            return "Please enter a number 1-53."
        
        csv_data = df.to_string(index=False)
        is_followup = False
        if history:
            matches = re.findall(r'Artwork ID (\d+)', str(history))
            if matches:
                art_id = int(matches[-1])
                is_followup = True

        if is_followup:
            row = df[df['ID'].astype(str).str.strip() == str(art_id)]
            if row.empty:
                return f"Could not find data for Artwork ID {art_id}."
            
            row = row.iloc[0]
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
                contents = [prompt]
                if img_bytes:
                    contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
                res = client.models.generate_content(model="gemini-3.6-flash", contents=contents)
                return res.text
            except Exception as e:
                traceback.print_exc()
                return f"Error: {str(e)}"
        else:
            prompt = f"""
            The user asked: "{user_text}"
            Here is the archival database metadata for all 53 artworks:
            {csv_data}
            Instructions: Answer the user's question using ONLY the database metadata provided above. List Artwork IDs and Titles. DO NOT HALLUCINATE.
            """
            try:
                res = client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
                return res.text
            except Exception as e:
                traceback.print_exc()
                return f"Error: {str(e)}"

    # SCENARIO B: Fresh request for a Single Artwork
    row = df[df['ID'].astype(str).str.strip() == str(art_id)]
    if row.empty:
        return f"Artwork ID {art_id} not found in the database."
        
    row = row.iloc[0]
    title = str(row.get('TITLE', 'Unknown'))
    artist = str(row.get('Artist (if known)', 'Unknown'))
    date = str(row.get('Date', 'Unknown'))
    style = str(row.get('Artistic style', 'Unknown'))
    source = str(row.get('Source', 'N/A'))
    csv_context = f"Title: {title}\nArtist: {artist}\nDate: {date}\nStyle: {style}\nSource: {source}"
    
    orig_path = get_image_path(art_id)
    seg_path = f"preloaded_segments/seg_{art_id}.jpg"
    col_path = f"preloaded_colors/colors_{art_id}.jpg"
    img_bytes = compress_image_for_gemini(orig_path) if orig_path else None

    prompt = f"""
    You are a strict, analytical art historian. You are analyzing Artwork ID {art_id}.
    Here is the EXACT archival data for this artwork:
    {csv_context}

    CRITICAL RULES (STRICTLY ENFORCED):
    1. YOU MUST NOT HALLUCINATE. Do not use any outside knowledge. If the archival data says 'Unknown' for the Artist, you MUST state "Artist Unknown". Do not invent names, dates, or historical facts not present in the archival data.
    2. YOUR RESPONSE MUST BE EXACTLY FOUR PARAGRAPHS. EACH PARAGRAPH MUST BE AT LEAST 150 WORDS. THE TOTAL RESPONSE MUST BE OVER 600 WORDS.
    3. Do not provide generic introductions or conclusions. Go directly into deep, analytical prose.

    PARAGRAPH STRUCTURE AND REQUIREMENTS:
    - Paragraph 1 (Archival & Contextual Analysis): Analyze the artwork using ONLY the archival data provided above. Discuss the title, artist (if known), date, artistic style, and source. Do not invent historical context; rely strictly on the provided metadata.
    - Paragraph 2 (Visual Analysis): Analyze the visual composition of the attached image. Discuss the mood, lighting, brushwork, and materiality based strictly on what you see in the attached image.
    - Paragraph 3 (Urban & Environmental Context): Relate the artwork to the urban and environmental history of Adelaide based strictly on the visual evidence (e.g., infrastructure, landscape, River Torrens, colonial settlement) and the archival date. Do not invent historical figures.
    - Paragraph 4 (Semantic Segmentation Analysis): You are looking at the original image. Conduct a rigorous textual analysis of how a semantic segmentation algorithm would break down this image. Discuss the distinct spatial regions, boundaries, and color fields (e.g., sky, water, land, architecture, figures). Explain what this computational breakdown reveals about the composition and spatial hierarchy of the artwork.
    """
    
    try:
        contents = [prompt]
        if img_bytes:
            contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
        res = client.models.generate_content(model="gemini-3.6-flash", contents=contents)
        res_text = res.text

        text_md = f"**Artwork ID {art_id}**\n\n{res_text}\n\n---\n"
        
        # Collect 3 image paths to send back to Gradio natively
        files_to_return = []
        if orig_path and os.path.exists(orig_path): files_to_return.append(orig_path)
        if os.path.exists(seg_path): files_to_return.append(seg_path)
        if os.path.exists(col_path): files_to_return.append(col_path)

        return {
            "text": text_md,
            "files": files_to_return
        }
    except Exception as e:
        traceback.print_exc()
        return f"Error generating response: {str(e)}"

# multimodal=False removes the upload button, making the UI much cleaner and faster
demo = gr.ChatInterface(fn=answer_question, title="Adelaide Artworks AI (1-53)", multimodal=False)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
```
