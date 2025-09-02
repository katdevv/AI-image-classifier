# AI Image Classifier

This is a simple **learning project** built with **Streamlit** and **TensorFlow (Keras)**.  
It lets you upload an image (JPG or PNG), and a pretrained **MobileNetV2** model (trained on ImageNet) will classify it into the top-3 most likely categories.  

---

## 🚀 Usage  

1. Create and activate a virtual environment:
```bash
   python3 -m venv venv
   source venv/bin/activate    # Linux/Mac
   .\venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

3. Add your OPENAI_API_KEY to a .env file:
```bash
   OPENAI_API_KEY=your_api_key_here
```

4. Run the app:
```bash
   streamlit run app.py
```

5. How to use
Upload an image (.jpg or .png).
Click Classify Image.
The app will display the top-3 predictions with confidence scores.