import os
import sys
import re
import json
import time
import io
import traceback

import pandas as pd
import gradio as gr
from google import genai
from google.genai import types
from PIL import Image

# ---------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------
client = genai.Client(api_key=os.environ.get("gemini_API_KEY"))

# Models are tried in order. Override with env vars if names change
# (run client.models.list() to see what your key can use).
LIVE_MODELS = os.environ.get("LIVE_MODELS", "gemini-2.5-flash,gemini-2.5-flash-lite").split(",")
WARM_MODELS = os.environ.get("WARM_MODELS", "gemini-2.5-pro,gemini-2.5-flash").split(",")

CACHE_FILE = "response_cache.json"
try:
    with open(CACHE_FILE, encoding="utf-8") as f:
        CACHE = json.load(f)
except Exception:
    CACHE = {}

def save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(CACHE, f, ensure_ascii=False, indent=1)

# ---------------------------------------------------------------
# 2. Load Excel data
# ---------------------------------------------------------------
try:
    df = pd.read_excel("data.xlsx")
    df.columns = df.columns.str.strip()
except Exception as e:
    print("Error loading data.xlsx:", e)
    df = pd.DataFrame()

# ---------------------------------------------------------------
# 3. Image lookup
# ---------------------------------------------------------------
IMAGE_LOOKUP = {}
for filename in os.listdir("."):
    if filename.split(".")[-1].lower() in ["jpg", "jpeg", "png", "webp"]:
        IMAGE_LOOKUP[filename.split(".")[0]] = filename

def get_image_path(image_id):
    return IMAGE_LOOKUP.get(str(image_id))

def compress_image_for_gemini(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=65)
        return buffer.getvalue()
    except Exception:
        with open(img_path, "rb") as f:
            return f.read()

# ---------------------------------------------------------------
# 4. Gemini helper with model fallback + retry
# ---------------------------------------------------------------
def call_gemini(contents, models):
    last_err = None
    for model in models:
        for attempt in range(2):
            try:
                res = client.models.generate_content(model=model.strip(), contents=contents)
                if res.text:
                    return res.text
            except Exception as e:
                last_err = e
                print(f"[{model}] attempt {attempt + 1} failed: {e}")
                time.sleep(3 * (attempt + 1))
    raise last_err if last_err else RuntimeError("Empty response")

# ---------------------------------------------------------------
# 5. Local fallback search (no API needed)
# ---------------------------------------------------------------
STOPWORDS = {"which", "what", "show", "list", "have", "with", "that", "there", "ones",
             "artworks", "artwork", "paintings", "painting", "images", "image", "any",
             "about", "does", "contain", "containing", "where", "find", "give", "tell"}

def local_search(user_text):
    words = [w for w in re.findall(r"[a-zA-Z]+", user_text.lower())
             if len(w) > 3 and w not in STOPWORDS]
    if not words:
        return "Please type an artwork number (1-53) or a keyword such as 'boat' or 'bridge'."
    hits = []
    for _, r in df.iterrows():
        blob = " ".join(str(v) for v in r.values).lower()
        if any(w in blob for w in words):
            hits.append(f"- **ID {r.get('ID')}**: {r.get('TITLE', 'Unknown')}")
    if not hits:
        return f"No artworks matched: {', '.join(words)}."
    return ("The AI service is busy, so this is a basic keyword search of the archive "
            f"for *{', '.join(words)}*:\n\n" + "\n".join(hits[:30]))

# ---------------------------------------------------------------
# 6. Main analysis generator (used live AND for cache warming)
# ---------------------------------------------------------------
def build_artwork_context(art_id):
    row = df[df["ID"].astype(str).str.strip() == str(art_id)]
    if row.empty:
        return None, None
    row = row.iloc[0]
    ctx = (f"Title: {row.get('TITLE', 'Unknown')}\n"
           f"Artist: {row.get('Artist (if known)', 'Unknown')}\n"
           f"Date: {row.get('Date', 'Unknown')}\n"
           f"Style: {row.get('Artistic style', 'Unknown')}\n"
           f"Source: {row.get('Source', 'N/A')}")
    return ctx, row

def generate_analysis(art_id, models):
    csv_context, _ = build_artwork_context(art_id)
    orig_path = get_image_path(art_id)
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
    contents = [prompt]
    if img_bytes:
        contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
    return call_gemini(contents, models)

# ---------------------------------------------------------------
# 7. Chat handler
# ---------------------------------------------------------------
def answer_question(user_text, history):
    if df.empty:
        return "Error loading data."

    user_text = (user_text or "").strip()
    if not user_text:
        return "Please enter a number 1-53."

    # Treat as an artwork request only if the message is short (e.g. "34", "show 34")
    # so follow-ups like "why does paragraph 2 say that?" aren't mistaken for IDs.
    art_id = None
    if len(user_text.split()) <= 4:
        for n in re.findall(r"\b(\d+)\b", user_text):
            if 1 <= int(n) <= 53:
                art_id = int(n)
                break

    # ---------- Path A: artwork number ----------
    if art_id:
        ctx, _ = build_artwork_context(art_id)
        if ctx is None:
            return f"Artwork ID {art_id} not found in the database."

        key = str(art_id)
        try:
            if key in CACHE:
                res_text = CACHE[key]
            else:
                res_text = generate_analysis(art_id, LIVE_MODELS)
                CACHE[key] = res_text
                save_cache()
        except Exception as e:
            traceback.print_exc()
            return (f"**Artwork ID {art_id}**\n\nThe AI service is busy right now. "
                    f"Archive details:\n\n{ctx}\n\nPlease try again in a minute.")

        orig_path = get_image_path(art_id)
        seg_path = f"preloaded_segments/seg_{art_id}.jpg"
        col_path = f"preloaded_colors/colors_{art_id}.jpg"
        files = [p for p in (orig_path, seg_path, col_path) if p and os.path.exists(p)]

        return {"text": f"**Artwork ID {art_id}**\n\n{res_text}\n\n---\n", "files": files}

    # ---------- Path B: free-text / follow-up ----------
    csv_data = df.to_string(index=False)
    last_id = None
    matches = re.findall(r"Artwork ID (\d+)", str(history))
    if matches:
        last_id = int(matches[-1])

    img_part = None
    if last_id:
        p = get_image_path(last_id)
        if p:
            img_part = types.Part.from_bytes(data=compress_image_for_gemini(p), mime_type="image/jpeg")

    prompt = f"""
    The user asked: "{user_text}"
    {"You are currently discussing Artwork ID " + str(last_id) + ". Its image is attached." if img_part else ""}
    The full database metadata for all 53 artworks is below:
    {csv_data}
    INSTRUCTIONS:
    - If the user is following up on or challenging the analysis of the current artwork, reply conversationally in under 150 words, based on the attached image.
    - If the user asks a GENERAL question (e.g. "which ones have boats?"), ignore the image and answer from the database metadata. List Artwork IDs and Titles.
    - YOU MUST NOT HALLUCINATE. Use only the provided data.
    """
    contents = [prompt] + ([img_part] if img_part else [])
    try:
        return call_gemini(contents, LIVE_MODELS)
    except Exception:
        traceback.print_exc()
        return local_search(user_text)

# ---------------------------------------------------------------
# 8. Cache warming:  python app.py warm
# ---------------------------------------------------------------
def warm_cache():
    for i in range(1, 54):
        if str(i) in CACHE:
            print("skip", i)
            continue
        try:
            CACHE[str(i)] = generate_analysis(i, WARM_MODELS)
            save_cache()
            print("done", i)
        except Exception as e:
            print("FAILED", i, e)
        time.sleep(10)  # stay under free-tier per-minute limits
    missing = [i for i in range(1, 54) if str(i) not in CACHE]
    print("Missing:", missing or "none. All 53 cached.")

# ---------------------------------------------------------------
# 9. Launch
# ---------------------------------------------------------------
demo = gr.ChatInterface(fn=answer_question, title="Adelaide Artworks AI (1-53)", multimodal=False)
demo.queue(default_concurrency_limit=10)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "warm":
        warm_cache()
    else:
        demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
