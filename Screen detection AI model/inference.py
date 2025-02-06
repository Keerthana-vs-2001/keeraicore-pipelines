import sys
import cv2
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from flask import Flask, request, jsonify
import os, os.path as os_path
import tempfile
app = Flask(__name__)
config_path =os.getenv("CONFIG_PATH","/app/config.yml")
model_path = os.getenv("MODEL_PATH","/app/model_final.pth")
#BASE_DIR = os.getcwd()
metadata1 = MetadataCatalog.get("my_dataset_train")
metadata1.thing_classes = ['web-elements', 'Button', 'Check box', 'Checked Radio button', 'Checked box', 'Dropdown box', 'Dropdown expand', 'Icon', 'Radio button', 'Scroll bar', 'Text box']
#BASE_DIR = os.getcwd()
cfg = get_cfg()
#change in path for docker--cfg.merge_from_file(os.path.join(BASE_DIR, "/content/config.yml"))
#cfg.MODEL.WEIGHTS = (os.path.join(BASE_DIR, "/content/model_final (2).pth"))
cfg.merge_from_file(config_path)
cfg.MODEL.WEIGHTS = model_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"
cfg.MODEL.WEIGHTS_ONLY = True
predictor = DefaultPredictor(cfg)
def processing_video(input_video_path, output_video_path):
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
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        outputs = predictor(frame)
        instances = outputs["instances"].to("cpu")
        pred_boxes = instances.pred_boxes
        pred_classes = instances.pred_classes
        pred_scores = instances.scores
        #noneedtodisplayscores
        for box, cls, score in zip(pred_boxes, pred_classes, pred_scores):
            box = box.tensor.numpy().flatten().astype(int)
            class_name= metadata1.thing_classes[cls.item()]
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, f'UI-element: {class_name}', (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        out.write(frame)
        frame_count +=1
        print(f"processed frame count {frame_count}")
        cap.release()
    out.release()
    print(f"Video processing completed successfully saved to {output_video_path}")
@app.route("/v1/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No file part"}), 400
    video_file = request.files['video']
    if video_file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    try:
      with tempfile.NamedTemporaryFile(delete=False) as tmp_input:
        input_video_path = tmp_input.name
        video_file.save(input_video_path)
        output_video_path = input_video_path.replace(".mp4", "_output.mp4")
        result = processing_video(input_video_path, output_video_path)
        if result != "Video processing completed successfully":
            return jsonify({"error": result}), 500
        return jsonify({"message": result,"output_video": output_video_path}), 200
      except Exception as e:
        return jsonify({"error":str(e)}),500
if __name__ == "__main__":
    #public_url = ngrok.connect(5000)
    #print(f"flask is running at{public_url}")
    app.run(debug=True, host='0.0.0.0', port=5000)
