import os
import pandas as pd
import gradio as gr
import re
import base64
import google.generativeai as genai
from PIL import Image, ImageDraw
import io
from skimage import segmentation, color
import numpy as np
from gtts import gTTS
from sklearn.cluster import KMeans
import concurrent.futures # THE SECRET TO 50 SECONDS!

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
            with open(img_path, "rb") as image_file:
                img_bytes = image_file.read()
            mime_type = {
                "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "webp": "image/webp"
            }.get(ext, "image/jpeg")
            return img_path, img_bytes, mime_type
    return None, None, "Error: Image file not found."

def optimize_image_for_gemini(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        byte_arr = io.BytesIO()
        img.save(byte_arr, format='JPEG', quality=70)
        return byte_arr.getvalue()
    except Exception as e:
        print(f"Optimize error: {e}")
        return img_bytes

def generate_segmentation_image(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.thumbnail((300, 300), Image.Resampling.LANCZOS)
        img_array = np.array(img)
        segments = segmentation.slic(img_array, n_segments=30, compactness=10, start_label=1, enforce_connectivity=False)
        segmented_img = color.label2rgb(segments, img_array, kind='avg', bg_label=0)
        seg_pil = Image.fromarray((segmented_img * 255).astype(np.uint8))
        return seg_pil
    except Exception as e:
        print(f"Segmentation error: {e}")
        return None

def extract_dominant_colors(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.thumbnail((50, 50))
        img_array = np.array(img).reshape((-1, 3))
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=1, max_iter=20)
        kmeans.fit(img_array)
        counts = np.unique(kmeans.labels_, return_counts=True)[1]
        percentages = (counts / counts.sum()) * 100
        colors = kmeans.cluster_centers_.astype(int)
        sorted_idx = np.argsort(-percentages)
        colors = colors[sorted_idx]
        percentages = percentages[sorted_idx]
        return colors, percentages
    except Exception as e:
        print(f"Color extraction error: {e}")
        return None, None

def create_color_bar(colors, percentages):
    try:
        bar_width = 800
        bar_height = 100
        color_bar = Image.new("RGB", (bar_width, bar_height))
        draw = ImageDraw.Draw(color_bar)
        x_offset = 0
        for i in range(len(colors)):
            w = int(bar_width * (percentages[i] / 100))
            r, g, b = colors[i]
            draw.rectangle([x_offset, 0, x_offset + w, bar_height], fill=(int(r), int(g), int(b)))
            x_offset += w
        return color_bar
    except Exception as e:
        print(f"Color bar error: {e}")
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
    except Exception as e:
        print(f"TTS Error: {e}")
        return ""

def pil_to_base64_html(img, filename, caption):
    if img is None: return ""
    try:
        buffered = io.BytesIO()
        img.convert("RGB").save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"""
        <div style="margin-top: 15px;">
            <p style="font-weight: bold; margin-bottom: 5px;">{caption} (Click image to download)</p>
            <a href="data:image/jpeg;base64,{img_str}" download="{filename}">
                <img src="data:image/jpeg;base64,{img_str}" alt="{caption}" style="width: 100%; border-radius: 8px; border: 1px solid #444; cursor: pointer;">
            </a>
        </div>
        """
    except Exception as e:
        print(f"Image to Base64 error: {e}")
        return ""

# 3. The Chat Engine
def answer_question(message, history):
    if df.empty: return {"text": "Error: Could not load data.xlsx."}

    user_text = message.get("text", "").strip()
    user_files = message.get("files", [])

    numbers = re.findall(r'\b(\d+)\b', user_text)
    requested_ids = []
    for num in numbers:
        if 1 <= int(num) <= 53: requested_ids.append(int(num))
    requested_ids = list(dict.fromkeys(requested_ids))

    is_followup = False
    if not requested_ids and history:
        history_str = str(history)
        matches = re.findall(r'Artwork ID (\d+)', history_str)
        if matches:
            requested_ids = [int(matches[-1])]
            is_followup = True

    if user_files:
        # (Upload Scenario logic remains the same to prevent breaking)
        parts = [f"User said: '{user_text}'. The user uploaded the following image(s) for you to analyze or compare." if user_text else "The user uploaded the following image(s) for you to analyze."]
        if requested_ids:
            for art_id in requested_ids:
                img_path, img_bytes, mime_type = get_image_data(art_id)
                if img_bytes:
                    parts.append(f"Reference Artwork ID {art_id}:")
                    parts.append({"mime_type": "image/jpeg", "data": optimize_image_for_gemini(img_bytes)})
        uploaded_img_bytes = None
        for file_path in user_files:
            ext = file_path.split('.')[-1].lower()
            mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
            with open(file_path, "rb") as f:
                uploaded_img_bytes = f.read()
                parts.append({"mime_type": mime, "data": optimize_image_for_gemini(uploaded_img_bytes)})
        try:
            response = model.generate_content(parts)
            res_text = response.text
            audio_html = text_to_speech_html(res_text, "uploaded_analysis.mp3")
            if uploaded_img_bytes:
                original_img = Image.open(io.BytesIO(uploaded_img_bytes))
                seg_img = generate_segmentation_image(uploaded_img_bytes)
                colors, pcts = extract_dominant_colors(uploaded_img_bytes)
                color_bar = create_color_bar(colors, pcts) if colors is not None else None
                text_md = f"**Analysis of Uploaded Image:**\n\n{res_text}\n\n---\n"
                text_md += pil_to_base64_html(original_img, "uploaded_original.jpg", "Your Uploaded Image")
                text_md += pil_to_base64_html(seg_img, "uploaded_segmentation.jpg", "Semantic Segmentation")
                text_md += pil_to_base64_html(color_bar, "uploaded_colors.jpg", "Dominant Color Palette")
                return {"text": text_md + audio_html}
            return {"text": res_text + audio_html}
        except Exception as e:
            return {"text": f"Error analyzing uploaded image: {str(e)}"}

    if len(requested_ids) > 1:
        # (Comparison Scenario logic remains the same)
        parts = [f"The user asked: '{user_text}'. Here are the requested artworks for you to compare/analyze:"]
        images_to_return = []
        for art_id in requested_ids:
            match_df = df[df['ID'].astype(str).str.strip() == str(art_id)]
            if not match_df.empty:
                row = match_df.iloc[0]
                title = str(row.get('TITLE', 'Unknown Title'))
                img_path, img_bytes, mime_type = get_image_data(art_id)
                if img_bytes:
                    parts.append(f"Artwork ID {art_id} ({title}):")
                    parts.append({"mime_type": "image/jpeg", "data": optimize_image_for_gemini(img_bytes)})
                    images_to_return.append((art_id, Image.open(io.BytesIO(img_bytes))))
        try:
            response = model.generate_content(parts)
            res_text = response.text
            audio_html = text_to_speech_html(res_text, "comparison_audio.mp3")
            text_md = f"**Comparison of Artworks {', '.join(map(str, requested_ids))}:**\n\n{res_text}\n\n---\n"
            for art_id, img in images_to_return:
                text_md += pil_to_base64_html(img, f"artwork_{art_id}.jpg", f"Artwork ID {art_id}")
            return {"text": text_md + audio_html}
        except Exception as e:
            return {"text": f"Error comparing artworks: {str(e)}"}

    if not requested_ids:
        # (General Question Scenario logic remains the same)
        csv_data = df.to_string(index=False)
        prompt = f"The user asked: '{user_text}'\nHere is the archival database metadata for all 53 artworks:\n{csv_data}\nInstructions: Answer the user's question based on the database metadata provided above. If they ask for a list of artworks with specific elements, provide the Artwork IDs and Titles from the text data."
        try:
            response = model.generate_content(prompt)
            res_text = response.text
            audio_html = text_to_speech_html(res_text, "general_answer.mp3")
            return {"text": res_text + audio_html}
        except Exception as e:
            return {"text": f"Error: {str(e)}"}

    # SCENARIO E: Fresh request for a Single Artwork (OPTIMIZED WITH PARALLEL PROCESSING)
    requested_id = requested_ids[0]
    match_df = df[df['ID'].astype(str).str.strip() == str(requested_id)]
    if match_df.empty: return {"text": f"Could not find data for Artwork ID {requested_id}."}

    row = match_df.iloc[0]
    title = str(row.get('TITLE', 'Unknown Title'))
    date = str(row.get('Date', 'Unknown Date'))
    artist = str(row.get('Artist (if known)', 'Unknown Artist'))
    style = str(row.get('Artistic style', 'Unknown Style'))
    img_path, img_bytes, mime_or_error = get_image_data(requested_id)
    if not img_bytes: return {"text": f"**Artwork ID {requested_id}:** {title} ({date}) by {artist}.\n\n*({mime_or_error})*"}

    csv_context = f"Title: {title}\nArtist: {artist}\nDate: {date}\nStyle: {style}\nSource: {row.get('Source', 'N/A')}"

    if is_followup:
        prompt = f"The user is asking a follow-up question or challenging your previous analysis about Artwork ID {requested_id} ({title}).\nArchival Data: {csv_context}\nUser's new input: '{user_text}'\nInstructions: Respond conversationally. Address their specific agreement, disagreement, or question directly based on the image provided. Do not repeat the original 4 paragraphs."
        try:
            response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": optimize_image_for_gemini(img_bytes)}])
            res_text = response.text
            audio_html = text_to_speech_html(res_text, f"artwork_{requested_id}_followup.mp3")
            return {"text": res_text + audio_html}
        except Exception as e:
            return {"text": f"Error: {str(e)}"}

    strict_prompt = f"""
    You are an expert art historian. The user requested information about Artwork ID {requested_id}.
    Archival data: {csv_context}
    RULES:
    1. YOUR RESPONSE MUST BE EXACTLY FOUR PARAGRAPHS AND NO MORE THAN 400 WORDS TOTAL.
    2. Paragraph 1: Introduce the artwork.
    3. Paragraph 2: Conduct a visual analysis of the attached image.
    4. Paragraph 3: Relate to urban history of Adelaide.
    5. Paragraph 4: Textual analysis of semantic segmentation (sky, water, land, etc.).
    """
    try:
        # START PARALLEL PROCESSING: Run Gemini, Segmentation, and Colors at the EXACT SAME TIME
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_gemini = executor.submit(model.generate_content, [strict_prompt, {"mime_type": "image/jpeg", "data": optimize_image_for_gemini(img_bytes)}])
            future_seg = executor.submit(generate_segmentation_image, img_bytes)
            future_colors = executor.submit(extract_dominant_colors, img_bytes)
            
            # Gather results
            response = future_gemini.result()
            response_text = response.text
            seg_img = future_seg.result()
            colors, pcts = future_colors.result()

        # Generate TTS and Image HTML as soon as the above finish
        color_bar = create_color_bar(colors, pcts) if colors is not None else None
        original_img = Image.open(io.BytesIO(img_bytes))

        text_md = f"**Artwork ID {requested_id}**\n\n{response_text}\n\n---\n"
        text_md += pil_to_base64_html(original_img, f"artwork_{requested_id}_original.jpg", "Original Artwork")
        text_md += pil_to_base64_html(seg_img, f"artwork_{requested_id}_segmentation.jpg", "Semantic Segmentation Map")
        text_md += pil_to_base64_html(color_bar, f"artwork_{requested_id}_colors.jpg", "Dominant Color Palette")

        audio_html = text_to_speech_html(response_text, f"artwork_{requested_id}_analysis.mp3")

        return {"text": text_md + audio_html}
    except Exception as e:
        return {"text": f"Error generating response: {str(e)}"}

# 4. Gradio Interface
demo = gr.ChatInterface(
    fn=answer_question,
    multimodal=True,
    title="Adelaide Artworks AI (1-53)",
    description="Ask about an artwork (1-53), compare multiple, ask general questions, or upload an image! (Click images to download)"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
