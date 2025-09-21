import cv2
import numpy as np
import json
import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import imutils

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ANSWER_KEY_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['ANSWER_KEY_FOLDER'] = ANSWER_KEY_FOLDER

# Create necessary directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(ANSWER_KEY_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def order_points(pts):
    """
    Sort the four corner points of the OMR sheet to ensure they are in a consistent order:
    top-left, top-right, bottom-right, and bottom-left.
    This is crucial for perspective transformation.
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left has the smallest sum
    rect[2] = pts[np.argmax(s)]  # Bottom-right has the largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right has the smallest difference
    rect[3] = pts[np.argmax(diff)]  # Bottom-left has the largest difference

    return rect

def get_answer_key(exam_set):
    """
    Load the answer key from a JSON file based on the exam set (A or B).
    """
    key_file = os.path.join(app.config['ANSWER_KEY_FOLDER'], f"answer_key.json")
    if not os.path.exists(key_file):
        raise FileNotFoundError(f"Answer key file not found: {key_file}")
    
    with open(key_file, 'r') as f:
        data = json.load(f)
        if exam_set in data:
            return data[exam_set]
        else:
            raise KeyError(f"Exam set '{exam_set}' not found in the answer key.")

def evaluate_omr_sheet(image_path, exam_set):
    """
    Main function to evaluate an OMR sheet using computer vision techniques.
    """
    # Load the image and convert it to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # Find contours in the edged image, keeping only the largest ones
    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    sheet_contour = None
    for c in contours:
        # Approximate the contour and check if it has 4 vertices (a rectangle)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            sheet_contour = approx
            break

    if sheet_contour is None:
        return {"error": "Could not find the OMR sheet outline. Please upload a clear image."}

    # Apply perspective transform to get a top-down view of the OMR sheet
    rect = order_points(sheet_contour.reshape(4, 2))
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(gray, M, (maxWidth, maxHeight))
    
    # Apply a fixed threshold to get a binary image
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Find all bubble contours on the rectified sheet
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    question_contours = []

    # Filter contours based on their size and aspect ratio to isolate bubbles
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
            question_contours.append(c)

    # Sort the bubbles from top to bottom
    question_contours = sorted(question_contours, key=lambda c: cv2.boundingRect(c)[1])

    # Get the answer key for the specified exam set
    try:
        answer_key = get_answer_key(exam_set)
    except (FileNotFoundError, KeyError) as e:
        return {"error": str(e)}

    # Group contours into questions (20 questions per subject, 5 subjects)
    questions = []
    subject_scores = [0, 0, 0, 0, 0] # Scores for each subject
    total_score = 0
    
    # Each question has 4 options, and there are 100 questions total
    questions = []
    for i in range(100):
        questions.append(question_contours[i*4:i*4+4])
        
    for i, q_cnts in enumerate(questions):
        # Sort bubbles for the current question from left to right
        q_cnts = sorted(q_cnts, key=lambda c: cv2.boundingRect(c)[0])
        marked_bubble = None
        
        # Determine which bubble is marked
        for j, c in enumerate(q_cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)

            # A marked bubble will have a high non-zero pixel count
            if total > 500: # Threshold value may need to be adjusted
                marked_bubble = j
                break
        
        # Check if the marked bubble matches the answer key
        subject_index = i // 20
        question_index_in_subject = i % 20
        
        correct_answer = answer_key[f"Subject {subject_index + 1}"][question_index_in_subject]
        
        if marked_bubble is not None and marked_bubble + 1 == correct_answer:
            subject_scores[subject_index] += 1
            total_score += 1
    
    results = {
        "total_score": total_score,
        "subject_scores": {
            "Subject 1": subject_scores[0],
            "Subject 2": subject_scores[1],
            "Subject 3": subject_scores[2],
            "Subject 4": subject_scores[3],
            "Subject 5": subject_scores[4],
        }
    }
    
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    exam_set = request.form.get('exam_set')
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename) and exam_set:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        results = evaluate_omr_sheet(filepath, exam_set)
        
        return jsonify(results)
    else:
        return jsonify({"error": "Invalid file type or exam set"}), 400

if __name__ == '__main__':
    app.run(debug=True)
