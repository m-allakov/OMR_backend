from flask import Flask, request, jsonify
import cv2 as cv
import numpy as np
import base64
import utilis  # Your existing utility functions

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_image():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    # Decode the base64 image
    image_data = base64.b64decode(data['image'])
    np_img = np.frombuffer(image_data, np.uint8)
    img = cv.imdecode(np_img, cv.IMREAD_COLOR)

    # Use your existing image processing code here
    # For example, you could use a function from utilis
    # img = utilis.process_image(img)

    # Encode the processed image back to base64
    _, buffer = cv.imencode('.jpg', img)
    processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'processedImageUrl': processed_image_base64})

if __name__ == '__main__':
    app.run(debug=True)
