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

# 1. Setup Google Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-3.5-flash')

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
                
            if ext in ["jpg", "jpeg"]:
                mime_type = "image/jpeg"
            elif ext == "png":
                mime_type = "image/png"
            elif ext == "webp":
                mime_type = "image/webp"
            else:
                mime_type = "image/jpeg"
                
            return img_path, img_bytes, mime_type
            
    return None, None, "Error: Image file not found."

def generate_segmentation_image(img_bytes):
    """Generates a semantic segmentation map, optimizing size for Render Free Tier"""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # OPTIMIZATION: Resize image to max 800px to prevent Render server from freezing
        max_size = (800, 800)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        img_array = np.array(img)
        
        # Apply SLIC segmentation
        segments = segmentation.slic(img_array, n_segments=100, compactness=10, start_label=1)
        segmented_img = color.label2rgb(segments, img_array, kind='avg', bg_label=0)
        
        seg_pil = Image.fromarray((segmented_img * 255).astype(np.uint8))
        byte_arr = io.BytesIO()
        seg_pil.save(byte_arr, format='JPEG')
        seg_bytes = byte_arr.getvalue()
        
        seg_base64 = base64.b64encode(seg_bytes).decode('utf-8')
        return seg_base64
    except Exception as e:
        print(f"Segmentation error: {e}")
        return None

# 3. The Chat Engine
def answer_question(user_prompt, history):
    if df.empty:
        return "Error: Could not load data.xlsx."
        
    user_prompt_lower = user_prompt.lower()
    numbers = re.findall(r'\b(\d+)\b', user_prompt_lower)
    
    matched_id_row = None
    requested_id = None
    
    for num in numbers:
        if 1 <= int(num) <= 48:
            match_df = df[df['ID'].astype(str).str.strip() == num]
            if not match_df.empty:
                matched_id_row = match_df.iloc[0]
                requested_id = num
                break
            
    if matched_id_row is not None:
        row = matched_id_row
        title = str(row.get('TITLE', 'Unknown Title'))
        date = str(row.get('Date', 'Unknown Date'))
        artist = str(row.get('Artist (if known)', 'Unknown Artist'))
        style = str(row.get('Artistic style', 'Unknown Style'))
        
        img_path, img_bytes, mime_or_error = get_image_data(requested_id)
        
        if not img_bytes:
            return f"**Artwork ID {requested_id}:** {title} ({date}) by {artist}.\n\n*({mime_or_error})*"

        mime_type = mime_or_error
        base64_img = base64.b64encode(img_bytes).decode('utf-8')
        
        csv_context = f"""
        Title: {title}
        Artist: {artist}
        Date: {date}
        Artistic Style/Medium: {style}
        Source: {row.get('Source', 'N/A')}
        """
        
        strict_prompt = f"""
        You are an expert art historian and computer vision analyst. The user requested information about Artwork ID {requested_id}.
        Here is the archival data for this artwork:
        {csv_context}
        
        RULES (DO NOT HALLUCINATE METADATA):
        1. YOUR RESPONSE MUST BE EXACTLY FOUR PARAGRAPHS.
        2. Paragraph 1: Introduce the artwork. State the exact Name, Artist, and Year. Provide a brief contextual background based on the archival data provided.
        3. Paragraph 2: Conduct a visual analysis of the attached image. Describe what you actually see (composition, colors, subjects, landscape, buildings).
        4. Paragraph 3: Explain how this artwork relates to the urban history of Adelaide as a city (e.g., colonial settlement, development of the River Torrens, infrastructure, or relations with Indigenous peoples).
        5. Paragraph 4: Conduct a textual analysis of the semantic segmentation of this artwork. Describe how the image can be broken down into semantic regions (e.g., sky, water, land, architecture, figures) and what these distinct segments represent in the context of the scene.
        """
        
        try:
            # Pass the raw image bytes directly to Gemini
            response = model.generate_content([
                strict_prompt,
                {"mime_type": mime_type, "data": img_bytes}
            ])
            response_text = response.text
            
            # Generate the Semantic Segmentation Image
            seg_base64 = generate_segmentation_image(img_bytes)
            
            # Format final output with 4 paragraphs and 2 images
            final_response = f"**Artwork ID {requested_id}**\n\n{response_text}\n\n"
            final_response += f"**Original Artwork:**\n![Artwork](data:{mime_type};base64,{base64_img})\n\n"
            
            if seg_base64:
                final_response += f"**Semantic Segmentation Map:**\n![Segmentation](data:image/jpeg;base64,{seg_base64})"
            else:
                final_response += "*(Semantic segmentation image could not be generated)*"
                
            return final_response
        except Exception as e:
            return f"Error generating response: {str(e)}"
            
    else:
        return "Please enter a valid artwork number between **1 and 48** to see the artwork, its metadata, and a visual analysis."

# 4. Gradio Interface
def torrens_chat(user_message, history):
    return answer_question(user_message, history)

demo = gr.ChatInterface(
    fn=torrens_chat,
    title="Adelaide Artworks AI (1-48)",
    description="Enter a number from 1 to 48 to view the artwork, receive a 4-paragraph analysis, and see semantic segmentation."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
