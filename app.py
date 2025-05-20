import streamlit as st
import os
from pathlib import Path
import subprocess
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="PCB Defect Detection - YOLO + DenseNet", layout="wide")
st.title("🔍 PCB Defect Detection using YOLO + DenseNet")

# Set paths
kaggle_dir = Path("/tmp/.kaggle")
dataset_dir = Path("/tmp/pcb_dataset/pcb-defect-dataset")
yaml_path = dataset_dir / "data.yaml"
model_path = Path("/tmp/pcb_yolo_densenet/yolo_pcb_defects/weights/best.pt")

# Step 1: Upload kaggle.json
st.header("Step 1: Upload kaggle.json")
kaggle_file = st.file_uploader("Upload your kaggle.json", type=["json"])
if kaggle_file:
    os.makedirs(kaggle_dir, exist_ok=True)
    with open(kaggle_dir / "kaggle.json", "wb") as f:
        f.write(kaggle_file.read())
    os.chmod(kaggle_dir / "kaggle.json", 0o600)
    os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_dir)
    st.success("✅ kaggle.json uploaded and saved")

    # Kaggle check
    try:
        subprocess.run(["kaggle", "--version"], check=True)
        st.success("✅ Kaggle CLI found!")
    except FileNotFoundError:
        st.error("❌ Kaggle CLI not found. Make sure 'packages.txt' has 'kaggle'.")

    # Download dataset
    if st.button("📥 Download PCB Dataset"):
        try:
            subprocess.run([
                "kaggle", "datasets", "download",
                "-d", "norbertelter/pcb-defect-dataset",
                "-p", str(dataset_dir.parent), "--unzip"
            ], check=True)
            st.success("✅ Dataset downloaded and extracted!")

            # Create data.yaml
            os.makedirs(dataset_dir, exist_ok=True)
            with open(yaml_path, "w") as f:
                f.write(f"""path: {dataset_dir.resolve()}
train: train/images
val: val/images
test: test/images

names:
  0: missing_hole
  1: mouse_bite
  2: open_circuit
  3: short
  4: spur
  5: spurious_copper
""")
            st.success("✅ data.yaml created")
        except Exception as e:
            st.error(f"❌ Dataset download failed: {e}")

# Step 2: Train YOLOv8
st.header("Step 2: Train YOLOv8 Model")
if st.button("🚀 Train YOLOv8"):
    if not yaml_path.exists():
        st.warning("⚠️ data.yaml not found")
    else:
        try:
            model = YOLO("yolov8s.pt")
            model.train(
                data=str(yaml_path),
                epochs=10,
                imgsz=640,
                batch=8,
                name="yolo_pcb_defects",
                project="/tmp/pcb_yolo_densenet",
                device=0 if torch.cuda.is_available() else 'cpu'
            )
            st.success("✅ YOLO Training Complete")
        except Exception as e:
            st.error(f"❌ Training failed: {e}")

# Step 3: Upload image and detect defects
st.header("Step 3: Upload Image for Detection")
uploaded_img = st.file_uploader("Upload a PCB image", type=["jpg", "jpeg", "png"])
if uploaded_img and model_path.exists():
    try:
        image = Image.open(uploaded_img).convert("RGB")
        model = YOLO(str(model_path))
        results = model.predict(source=image, conf=0.25)
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()

        boxes = results[0].boxes
        if boxes:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                draw.rectangle([(x1, y1), (x2, y2)], outline="white", width=3)
                draw.text((x1, y1 - 20), label, font=font, fill="white")
            st.image(image, caption="🔍 Detected Defects", use_column_width=False)
        else:
            st.warning("❌ No defects detected.")
    except Exception as e:
        st.error(f"❌ Detection failed: {e}")
elif uploaded_img:
    st.warning("⚠️ Please train the model before running detection.")
