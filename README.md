# UI_Annotation 🚀

**UI_Annotation** is a Python-based graphical tool designed to simplify the annotation of AI datasets using **bounding boxes (BBoxes)**.  
It provides an **interactive interface** to visualize datasets, annotate images efficiently, and leverage **machine vision** or **pretrained AI models** to assist with annotations.

---

## Features ✨

- **📂 Dataset Navigation:** Easily browse through images in a folder or dataset.  
- **🖌 Bounding Box Annotation:** Draw, edit, and delete bounding boxes for objects in images.  
- **📊 Ground Truth & Predictions:** Visualize ground truth (GT) boxes and AI predictions side by side.  
- **🎛 Transparency & Visibility:** Adjust visibility and transparency of GT, Predicted, and Extra boxes with sliders.  
- **🛠 Modes & Tools:** Switch between annotation modes for precise editing.  
- **🤖 Machine Vision Assistance:** Use pre-trained models or computer vision algorithms to propose bounding boxes automatically.  
- **🔍 Pan & Zoom:** Navigate large images with arrow-based panning and zoom options.  
- **💾 Save & Export:** Save annotations in standard formats for training AI models.  

---
## Usage 🖥

### 1. Launch the UI annotation script
Running the script should open the interface like this:

<img width="1287" height="956" alt="UI Annotation Screenshot" src="https://github.com/user-attachments/assets/8b6a218d-8083-4167-ae22-3c4afd6b9b26" />

---

### 2. Select Dataset
Select a dataset folder using the **"Select Parent Folder"** button (top LEFT).  
- The dataset should be in **YOLO format** (images + labels folder).  
- If you only have images, create a **fake labels folder**.

---

### 3. Navigate Images
- Use the **slider** or **arrow buttons** on top to browse images.  
- Toggle **visibility** and adjust **transparency** of GT, Pred, and Extra boxes.

---

### 4. Annotation Modes
Use the modes to perform actions on your images. These modes require your action to take effect:

1. **Draw** ✏️: Draw a new BBox  
2. **Move/Resize** ✂️: Modify the current BBox  
3. **Delete** 🗑️: Permanently remove a BBox  
4. **Validate** ✅: Transfer BBox from prediction to GT

---

### 5. Use AI Prediction
If you have a trained YOLO model, you can annotate images as follows:

1. **Import Weights:** `best.pt`  
2. **Click on Predict Image**  
3. **Visualize Predictions:** Tick the checkboxes on the left  
4. **Transfer Predictions:** If correct, move predicted BBoxes to GT

---

### 6. Use Machine Vision Tools
Currently supported tools for assisting annotation:

- **Excessive Green Detection** 🌿  
- **Shape Detection:** Circle, Triangle, Rectangle, Polygon

---

### 7. To Come Next ⏳
- Currently, the tool only supports **single-class annotation**.  
- **Multi-class annotation** is coming soon.
