import streamlit as st
from PIL import Image
from ultralyticsplus import YOLO, postprocess_classify_output

# Load model
model = YOLO('best.pt')

# Set model parameters
model.overrides['conf'] = 0.25  # Model confidence threshold

# Streamlit app
st.title("Chest X-ray Classification")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Perform inference
    results = model.predict(image)

    # Process results
    processed_result = postprocess_classify_output(model, result=results[0])
    # st.write(processed_result)

    if processed_result["PNEUMONIA"] > processed_result["NORMAL"]:
        st.write("Результат: PNEUMONIA")
    else:
        st.write("Результат: NORMAL")
    # st.write("Processed Result:")