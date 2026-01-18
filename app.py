import os
import cv2
import easyocr
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for
from ultralytics import YOLO
from datetime import datetime
from sort import Sort

# Initialize Flask app
app = Flask(__name__)

# Folders for uploads and screenshots
UPLOAD_FOLDER = "static/uploads"
SCREENSHOT_FOLDER = "static/screenshots"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SCREENSHOT_FOLDER"] = SCREENSHOT_FOLDER

# Load YOLO model
MODEL_PATH = "yolo-weights/YOLOV8N-V9.pt"
CONFIDENCE_THRESHOLD = 0.1
NUMBER_PLATE_CONFIDENCE_THRESHOLD = 0.1

# Initialize EasyOCR reader and YOLO model
reader = easyocr.Reader(['en'])
model = YOLO(MODEL_PATH)

# Class labels and colors
classNames = ['number plate', 'rider', 'with helmet', 'without helmet']
classColors = {
    'number plate': (255, 0, 0),
    'rider': (0, 255, 255),
    'with helmet': (0, 255, 0),
    'without helmet': (0, 0, 255)
}

# Global variable to track uploaded video path
video_path = None

# Initialize SORT tracker
tracker = Sort()


def save_screenshot(img):
    """Save a screenshot to the screenshots folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_path = os.path.join(app.config["SCREENSHOT_FOLDER"], f"screenshot_{timestamp}.jpg")
    cv2.imwrite(screenshot_path, img)
    print(f"Screenshot saved: {screenshot_path}")


def detect_objects(source):
    """Object detection and tracking from a video source."""
    cap = cv2.VideoCapture(source)
    tracked_riders = set()

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = cv2.resize(img, (800, 600))
        results = model(img, stream=True)

        rider_detections = []
        without_helmet_detected = False
        number_plate_boxes = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                currentClass = classNames[cls] if cls < len(classNames) else "Unknown"

                # Confidence threshold based on class type
                min_confidence = NUMBER_PLATE_CONFIDENCE_THRESHOLD if currentClass == 'number plate' else CONFIDENCE_THRESHOLD

                if conf >= min_confidence:
                    # Store rider and number plate detections
                    if currentClass == 'rider':
                        rider_detections.append([x1, y1, x2, y2, conf])

                    if currentClass == 'without helmet':
                        without_helmet_detected = True

                    if currentClass == 'number plate':
                        number_plate_boxes.append((x1, y1, x2, y2))

                    # Draw bounding box and label
                    color = classColors.get(currentClass, (255, 255, 255))

                    # Draw the bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                    # Get text size for background dimensions
                    (text_width, text_height), _ = cv2.getTextSize(f'{currentClass}', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                    # Draw black background rectangle behind text
                    cv2.rectangle(img, (x1, y1 - text_height - 5), (x1 + text_width + 10, y1), (0, 0, 0), -1)

                    # Draw the class name on the black background
                    cv2.putText(img, f'{currentClass}', (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Update SORT tracker
        rider_detections = np.array(rider_detections) if rider_detections else np.empty((0, 5))
        track_results = tracker.update(rider_detections)

        # Count unique riders
        for track in track_results:
            _, _, _, _, track_id = map(int, track)
            tracked_riders.add(track_id)

        rider_count = len(tracked_riders)

        # Magnify and OCR number plates if "without helmet" is detected
        if without_helmet_detected:
            for (x1, y1, x2, y2) in number_plate_boxes:
                plate_crop = img[y1:y2, x1:x2]
                if plate_crop.size > 0:
                    # Magnify number plate (resize 2x)
                    magnified_plate = cv2.resize(plate_crop, (0, 0), fx=2, fy=2)

                    # Position magnified plate above original
                    mag_x1 = x1
                    mag_y1 = max(0, y1 - magnified_plate.shape[0] - 10)
                    mag_x2 = mag_x1 + magnified_plate.shape[1]
                    mag_y2 = mag_y1 + magnified_plate.shape[0]

                    # Ensure overlay stays within frame bounds
                    if mag_x2 <= img.shape[1] and mag_y2 <= img.shape[0]:
                        img[mag_y1:mag_y2, mag_x1:mag_x2] = magnified_plate

                    # Perform OCR on the cropped plate
                    plate_text = reader.readtext(plate_crop, detail=0)
                    for plate in plate_text:
                        print(f"Detected Plate: {plate}")
                        save_screenshot(img)

        # Display total rider count
        # Get text size for background dimensions
        (text_width, text_height), _ = cv2.getTextSize(f'Total Riders: {rider_count}', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

        # Draw black background rectangle behind the text
        cv2.rectangle(img, (10, 5), (10 + text_width + 10, 40), (0, 0, 0), -1)

        # Draw the 'Total Riders' text on the black background in white color
        cv2.putText(img, f'Total Riders: {rider_count}', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Stream frame
        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads."""
    global video_path
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    video_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(video_path)

    # Reset tracker on new upload
    tracker.__init__()

    return redirect(url_for('play_video'))


@app.route('/play')
def play_video():
    """Render the video playback page."""
    global video_path
    if not video_path:
        return redirect(url_for('index'))
    return render_template('play.html', video_url=video_path)


@app.route('/video_feed')
def video_feed():
    """Stream uploaded video."""
    global video_path
    if not video_path:
        return redirect(url_for('index'))
    return Response(detect_objects(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/live_feed')
def live_feed():
    """Stream live webcam feed."""
    return Response(detect_objects(0), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/image_upload', methods=['POST'])
def image_upload():
    """Handle image uploads."""
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(image_path)

    for _ in detect_objects(image_path):
        pass

    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
