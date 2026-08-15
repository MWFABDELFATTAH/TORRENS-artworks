import os
import pandas as pd
from groq import Groq
import gradio as gr
import re
import base64

# 1. Setup Groq (Cloud AI)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# 2. Load Excel Data
try:
    df = pd.read_excel("data.xlsx")
    df.columns = df.columns.str.strip() # Clean column names
except Exception as e:
    print(f"Error loading Excel: {e}")
    df = pd.DataFrame()

def get_image_path_and_base64(image_id):
    """Searches the images folder for a file starting with the ID, regardless of extension."""
    if not os.path.exists("images"):
        return None, None
        
    for filename in os.listdir("images"):
        # Check if file starts with "1." or "2." etc.
        if filename.startswith(f"{image_id}."):
            img_path = os.path.join("images", filename)
            with open(img_path, "rb") as image_file:
                base64_img = base64.b64encode(image_file.read()).decode('utf-8')
                
            # Determine mime type for base64 string
            ext = filename.split('.')[-1].lower()
            if ext == "jpg" or ext == "jpeg":
                mime_type = "image/jpeg"
            elif ext == "png":
                mime_type = "image/png"
            elif ext == "webp":
                mime_type = "image/webp"
            else:
                mime_type = "image/jpeg" # default
                
            return img_path, (base64_img, mime_type)
            
    return None, None

# 3. The Chat Engine
def answer_question(user_prompt, history):
    if df.empty:
        return "Error: Could not load data.xlsx."
        
    user_prompt_lower = user_prompt.lower()
    
    # Extract ALL numbers from user input
    numbers = re.findall(r'\b(\d+)\b', user_prompt_lower)
    
    matched_id_row = None
    requested_id = None
    
    for num in numbers:
        # Only look for IDs between 1 and 48
        if 1 <= int(num) <= 48:
            match_df = df[df['ID'].astype(str).str.strip() == num]
            if not match_df.empty:
                matched_id_row = match_df.iloc[0]
                requested_id = num
                break
            
    # --- USER TYPED A SPECIFIC ID NUMBER (e.g., "5") ---
    if matched_id_row is not None:
        row = matched_id_row
        
        # Extract metadata safely
        title = str(row.get('TITLE', 'Unknown Title'))
        date = str(row.get('Date', 'Unknown Date'))
        artist = str(row.get('Artist (if known)', 'Unknown Artist'))
        style = str(row.get('Artistic style', 'Unknown Style'))
        
        # Find image dynamically
        img_path, image_data = get_image_path_and_base64(requested_id)
        
        if not image_data:
            return f"**Artwork ID {requested_id}:** {title} ({date}) by {artist}.\n\n*(Error: Image file for ID {requested_id} is missing in the 'images' folder.)*"

        base64_img, mime_type = image_data
        
        # Context for Paragraph 1
        csv_context = f"""
        Title: {title}
        Artist: {artist}
        Date: {date}
        Artistic Style/Medium: {style}
        Source: {row.get('Source', 'N/A')}
        """
        
        # Using Llama 3.2 Vision to actually analyze the image
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
            # Groq Vision API Call
            res = client.chat.completions.create(
                model="llama-3.2-90b-vision-preview",
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
            
            # Format final output to include the image at the bottom
            final_response = f"**Artwork ID {requested_id}**\n\n{response_text}\n\n![Artwork](data:{mime_type};base64,{base64_img})"
                
            return final_response
        except Exception as e:
            return f"Error generating response: {str(e)}"
            
    # --- USER TYPED A KEYWORD OR INVALID NUMBER ---
    else:
        return "Please enter a valid artwork number between **1 and 48** to see the artwork, its metadata, and a visual analysis."

# 4. Gradio Interface
def torrens_chat(user_message, history):
    return answer_question(user_message, history)

# Note: 'theme=gr.themes.Soft()' has been removed to fix the Render crash
demo = gr.ChatInterface(
    fn=torrens_chat,
    title="Adelaide Artworks AI (1-48)",
    description="Enter a number from 1 to 48 to view the artwork and receive a two-paragraph historical and visual analysis."
)

# Bind to port for Render deployment
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
