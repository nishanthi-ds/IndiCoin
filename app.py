import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import ultralytics

# Apply light blue gradient background

# Apply light yellow background
light_yellow_style = """
<style>
    .stApp {
        background-color: #FFFACD;  /* Light yellow background */
        color: black;  /* Set default text color to black */
    }

    html, body, [class*="css"] {
        color: black;
        font-weight: normal;
    }

    .stButton > button {
        background-color: #f4c542;  /* Yellow-orange button */
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }

    .stSelectbox label {
        color: black !important;
    }

    h1, h2, h3, h4, h5, h6, p, label {
        color: black !important;
    }
</style>
"""

# Apply the custom style
st.markdown(light_yellow_style, unsafe_allow_html=True)



# Function to detect coins in the uploaded image
def detect(img_path):
    model = ultralytics.YOLO('best.pt', task='detect')

    labels = model.names

    def resize_image(image, target_size):
        h, w = image.shape[:2]
        if w < target_size[0] or h < target_size[1]:
            interpolation = cv2.INTER_CUBIC  # Good for upscaling
        else:
            interpolation = cv2.INTER_AREA  # Good for downscaling
        resized_image = cv2.resize(image, target_size, interpolation=interpolation)
        return resized_image

    bbox_colors = [(0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0), (88, 159, 106),
                   (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)]

    lab = {0: '50 paise ', 1: ' ₹1 ', 2: ' ₹2 ', 3: ' ₹5 ', 4: ' ₹10 '}
    coin_lab = {0: 0.5, 1: 1, 2: 2, 3: 5, 4: 10}

    frame = cv2.imread(img_path)
    frame = cv2.resize(frame, (750, 750))

    # Run inference on frame
    results = model(frame, iou=0.2, verbose=False)

    # Extract results
    detections = results[0].boxes

    object_count = 0
    coins_count = 0

    for i in range(len(detections)):
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        classidx = int(detections[i].cls.item())
        new_lab = {0: 1, 1: 4, 2: 2, 3: 3, 4: 0}
        predicted_label = new_lab[classidx]

        conf = detections[i].conf.item()
        classname = lab[predicted_label]

        if conf > 0.5:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 5)

            label = f'{classname}: {int(conf * 100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), color,
                          cv2.FILLED)

            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image_pil)
            font = ImageFont.truetype("arial.ttf", 16)
            draw.text((xmin, label_ymin - 20), label, font=font, fill=(0, 0, 0))

            object_count += 1
            frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        coins_count += coin_lab[predicted_label]

    # Display detection results
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype("arial.ttf", 24)
    draw.text((10, 60), f'Total Amount ₹{coins_count}', font=font, fill=(0, 255, 0))
    frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    cv2.putText(frame, f'Number of Coins: {object_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
    frame = resize_image(frame, (750, 750))
    cv2.imwrite('out.jpg', frame)

    return frame, coins_count,object_count

def correct_image_orientation(image):
    try:
        # Get the EXIF data
        exif = image._getexif()

        if exif is not None:
            for tag, value in exif.items():
                if tag == 274:  # Orientation tag in EXIF data
                    if value == 3:
                        image = image.rotate(180, expand=True)
                    elif value == 6:
                        image = image.rotate(270, expand=True)
                    elif value == 8:
                        image = image.rotate(90, expand=True)
                    break
    except (AttributeError, KeyError, IndexError):
        # In case there's no EXIF data or an error occurs, just return the original image
        pass
    return image

# Custom CSS for colorful drag-and-drop area
st.markdown("""
    <style>
    .stFileUploader {
        background: linear-gradient(135deg, #ff6f61, #ffbc42);
        border: 2px dashed #fff;
        border-radius: 10px;
        padding: 25px;
        text-align: center;
        font-size: 18px;
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stFileUploader:hover {
        background: linear-gradient(135deg, #ffbc42, #ff6f61);
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit interface
st.title("Coin Detection and Counting")

coin_image = Image.open("demo.png")  # Provide the path to your simple coin image
coin_image = coin_image.resize((700, 300))
st.subheader("Demo")
st.image(coin_image, caption="Coin Image", use_container_width=True)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])



if uploaded_file is not None:
    # Convert uploaded file to image
    img = Image.open(uploaded_file)


    # Save the image to a temporary file for detection
    img_path = "temp_image.jpg"
    img = correct_image_orientation(img)
    img.save(img_path)

    # Call the detection function
    output_img ,amount,coins = detect(img_path)

    # Convert output image to displayable format
    output_img_pil= Image.fromarray(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    output_img_pil = output_img_pil.resize((400, 400))
    st.subheader("Detected Output")

    # Display the result image
    st.image(output_img_pil)

    # Example values (replace with actual detection results)
    total_amount = 28
    total_coins = 5

    # Show Total Coins and Amount in red bold text
    st.markdown(f"""
        <div style='color:red; font-size:22px; font-weight:bold;'>
            🪙 Total Coins Detected: {coins}<br>
            💰 Total Amount: ₹{amount}
        </div>
    """, unsafe_allow_html=True)
