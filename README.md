Adelaide Artworks AI Chatbot

A Gradio-based chatbot powered by Groq's Llama 3.2 Vision model that analyzes 48 historical artworks related to Adelaide and the River Torrens.



How to Run Locally

Ensure you have Python 3.10+ installed.

Install dependencies: pip install -r requirements.txt

Set your Groq API key in your terminal:

Mac/Linux: export GROQ\_API\_KEY="your\_api\_key\_here"

Windows: set GROQ\_API\_KEY="your\_api\_key\_here"

Place data.xlsx in the root folder.

Place your 48 images in an images/ folder named 1.jpg through 48.jpg.

Run the app: python app.py

Deployment

This app is configured for deployment on Render.

