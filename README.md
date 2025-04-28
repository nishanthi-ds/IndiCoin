
# IndiCoin: Automated Coin Value Estimator

This project uses YOLO to detect and estimate the value of coins from images.

## Project Workflow

1. **Prepare Dataset**
   - Organize images and annotations.
   - Create `data.yaml` file with dataset paths.

2. **Train/Test Split**
   - Split the dataset into training and testing sets.

3. **Train YOLO Model**
   - Command used:
     ```bash
     yolo detect train data=/content/data.yaml model=yolo11s.pt epochs=30 imgsz=640
     ```

4. **Model Saving**
   - Trained model is saved automatically after training.

5. **Evaluation & Testing**
   - Evaluate model performance on the test set.

6. **Model Inference**
   - Use trained model to detect and estimate coin values on new images.

## Folder Structure

```
/content
  /data.yaml
  /train_images
  /test_images
  /models
  /results
```

## Requirements

- Python
- YOLOv8 (Ultralytics)
- Torch
- OpenCV

Install requirements:
```bash
pip install ultralytics opencv-python torch
```

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/indicoin-coin-estimator.git
   ```

2. Navigate into the project directory:
   ```bash
   cd indicoin-coin-estimator
   ```

3. Start training:
   ```bash
   yolo detect train data=/content/data.yaml model=yolo11s.pt epochs=30 imgsz=640
   ```

4. Perform inference/testing after training.

## Notes

- Ensure correct paths in `data.yaml`.
- Modify epochs, model size (`imgsz`), and model architecture as needed.

---

Created with ❤️ by [Your Name]
