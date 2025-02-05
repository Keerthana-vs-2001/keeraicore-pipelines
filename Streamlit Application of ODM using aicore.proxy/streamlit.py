import sys
import cv2
import easyocr
import streamlit as st
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from flask import Flask, request, jsonify
import os, os.path as os_path
import tempfile
import time
st.write("starting app")
ocr_reader = easyocr.Reader(['en'])
#app = Flask(__name__)
metadata1 = MetadataCatalog.get("my_dataset_train")
metadata1.thing_classes = ['web-elements', 'Button', 'Check box', 'Checked Radio button', 'Checked box', 'Dropdown box', 'Dropdown expand', 'Icon', 'Radio button', 'Scroll bar', 'Text box']
BASE_DIR = os.getcwd()
cfg = get_cfg()
cfg.MODEL.WEIGHTS_ONLY = True
cfg.merge_from_file(os.path.join(BASE_DIR, "/content/config.yml"))
cfg.MODEL.WEIGHTS = (os.path.join(BASE_DIR, "/content/model_final (2).pth"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu" # if not torch.cuda.is_available() else "cuda"
cfg.MODEL.WEIGHTS_ONLY = True
predictor = DefaultPredictor(cfg)
st.write("loaded model")
def extract_text(image,box):
    x1, y1,x2,y2 = box
    cropped =image[y1:y2, x1:x2]
    results = ocr_reader.readtext(cropped,detail =0)
    return " ".join(results) if results else None
def generate_description(context,question):
  AI_API_URL= "https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/d5d922416b1fcffc/chat/completions?api-version=2023-05-15"
  API_TOKEN = "https://genai-ltim.authentication.eu10.hana.ondemand.com"
  headers ={
      "AI-Resource-Group": "default",
      "Content-Type": "application/json",
      "Authorization": f"Bearer https://genai-ltim.authentication.eu10.hana.ondemand.com"
  }
  prompt =f"Answer the question based on the following context onlt :\n{context}\n\nQuestion:{question}"
  response =response.post(AI_API_URL,headers = headers,json={"input":prompt})
  if response.status_code ==200:
    return response.json().get("generated_text","No description generated")
  else:
    return f"Failed to generate description:{response.text}"




def processing_video(input_video_path,output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video file")
        sys.exit(1)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    frame_count = 0
    description =[]
    last_prediction_time = 0
    prediction_interval = 1.00
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_time =time.time()
        if current_time - last_prediction_time >= prediction_interval:
          outputs = predictor(frame)
          instances = outputs["instances"].to("cpu")
          pred_boxes = instances.pred_boxes
          pred_classes = instances.pred_classes
          pred_scores = instances.scores
          detected_elements =[]
          for box, cls, score in zip(pred_boxes, pred_classes, pred_scores):
             box = box.cpu().numpy().flatten().astype(int)
             class_name= metadata1.thing_classes[cls.item()]
             extracted_text = extract_text(frame,box)
             if extracted_text:
                description = f"{class_name} with text:{extracted_text}"
             else:
                 description = f"{class_name} (No text detected)"
             detected_elements.append( description)
             cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
             cv2.putText(frame, f'ui-element: {class_name}', (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
          last_prediction_time = current_time
          out.write(frame)
          frame_count +=1
          print(f"processed frame count {frame_count}")
          cap.release()
    out.release()
    context ="\n".join(detected_elements)
    question = "Describe what is happening on the screen"
    description =generate_description(context,question)
    return description,frame

    return "Video processing completed successfully"
#@app.route("/v1/predict", methods=["POST"])
def main():
 st.title("video processing description generator")
 uploaded_file = st.file_uploader("Upload a video", type=["mp4","avi","mov"])
 if uploaded_file is not None:
   with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
        input_video_path = tmp_input.name
        uploaded_file.save(input_video_path)
        st.video(uploaded_file)
        st.write("processing the video...")
        description,processed_frame =processing_video(input_video_path)
        st.write("Description of screen ")
        st.write(description)
        if processed_frame is not None:
           st.image(processed_frame,caption="Processed Frame",use_column_width=True)
 os.system("streamlit is running")
 if __name__ == "__main__":
    main()

