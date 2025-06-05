import streamlit as st
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import ViTImageProcessor, ViTForImageClassification
import torch

# Load model and processor
@st.cache_resource
def load_model():
    model_name = "google/vit-base-patch16-224"
    model = ViTForImageClassification.from_pretrained(model_name)
    processor = ViTFeatureExtractor.from_pretrained(model_name)
    return model, processor

# Prediction function
def classify(image, model, processor):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    top5_probs, top5_indices = torch.topk(probs.squeeze(), k=5)
    return [
        (model.config.id2label[idx.item()], round(prob.item() * 100, 2))
        for idx, prob in zip(top5_indices, top5_probs)
    ]


# Streamlit UI
st.title("🖼️ VisionSnap Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    # st.image(image, caption="Uploaded Image", use_column_width=True)
    st.image(image, caption="Uploaded Image", use_container_width=True)


    model, processor = load_model()
    predictions = classify(image, model, processor)

    st.subheader("Top 5 Predictions:")
    for label, score in predictions:
        st.write(f"{label}: {score}%")

    st.write(f"**Image Format:** {image.format or 'N/A'}, **Size:** {image.size}, **Mode:** {image.mode}")

    # Download button
    result_str = "\n".join(f"{label}: {score}%" for label, score in predictions)
    st.download_button("Download Prediction", result_str, file_name="result.txt")
