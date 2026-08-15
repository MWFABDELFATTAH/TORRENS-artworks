import os
import pandas as pd
from openai import OpenAI
import gradio as gr
import re
import base64

# 1. Setup Google Gemini using the reliable OpenAI SDK
client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# 2. Load Excel Data
try:
    df = pd.read_excel("data.xlsx")
    df.columns = df.columns.str.strip()
except Exception as e:
    print(f"Error loading Excel: {e}")
    df = pd.DataFrame()

def get_image_path_and_base64(image_id):
    img_dir = "."
        
    for filename in os.listdir(img_dir):
        ext = filename.split('.')[-1].lower()
        if filename.lower().startswith(f"{image_id}.") and ext in ["jpg", "jpeg", "png", "webp"]:
            img_path = os.path.join(img_dir, filename)
            with open(img_path, "rb") as image_file:
                base64_img = base64.b64encode(image_file.read()).decode('utf-8')
                
            if ext in ["jpg", "jpeg"]:
                mime_type = "image/jpeg"
            elif ext == "png":
                mime_type = "image/png"
            elif ext == "webp":
                mime_type = "image/webp"
            else:
                mime_type = "image/jpeg"
                
            return img_path, (base64_img, mime_type)
            
    available_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))][:10]
    return None, f"Error: Could not find a file starting with '{image_id}.' in the root folder. First 10 image files found: {available_files}"

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
        
        img_path, image_data = get_image_path_and_base64(requested_id)
        
        if isinstance(image_data, str):
            return f"**Artwork ID {requested_id}:** {title} ({date}) by {artist}.\n\n*({image_data})*"

        base64_img, mime_type = image_data
        
        csv_context = f"""
        Title: {title}
        Artist: {artist}
        Date: {date}
        Artistic Style/Medium: {style}
        Source: {row.get('Source', 'N/A')}
        """
        
        strict_prompt = f"""
        You are an expert art historian. The user requested information about Artwork ID {requested_id}.
        Here is the archival data for this artwork:
        {csv_context}
        
        RULES (DO NOT HALLUCINATE METADATA):
        1. YOUR RESPONSE MUST BE EXACTLY TWO PARAGRAPHS.
        2. Paragraph 1: Introduce the artwork. State the exact Name, Artist, and Year. Provide a brief contextual background based on the archival data provided.
        3. Paragraph 2: Conduct a visual analysis of the attached image. Describe what you actually see (composition, colors, subjects, landscape, buildings). Then, relate this visual evidence to the urban history of Adelaide as a city (e.g., colonial settlement, development of the River Torrens, infrastructure, or relations with Indigenous peoples).
        """
        
        try:
            # Using OpenAI SDK to call Gemini 1.5 Flash
            res = client.chat.completions.create(
                model="gemini-1.5-flash",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": strict_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_img}"
                                }
                            }
                        ]
                    }
                ]
            )
            response_text = res.choices[0].message.content
            
            final_response = f"**Artwork ID {requested_id}**\n\n{response_text}\n\n![Artwork](data:{mime_type};base64,{base64_img})"
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
    description="Enter a number from 1 to 48 to view the artwork and receive a two-paragraph historical and visual analysis."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
