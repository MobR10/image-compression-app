from flask import Flask, send_file, request
from flask_cors import CORS
from colocviu import process_image, image_to_bytes

app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:5500"])

@app.route('/processImage', methods=['POST'])
def process_image_route():
    factor = int(request.form.get("factor", 50))
    radius = int(request.form.get("radius", 2))
    file = request.files.get("image")

    if not file:
        return {"error": "No file uploaded"}, 400

    file_bytes = file.read()
    out = process_image(file_bytes, factor, radius)
    out_bytes = image_to_bytes(out)

    return send_file(out_bytes, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(debug=True)
