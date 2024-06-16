# app/server.py
from flask import Flask, render_template, request, jsonify, send_from_directory

from infrared.parser import process_input

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename)


@app.route('/process_file', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    smiles = request.form.get('smiles')
    is_abs = request.form.get('is_abs')

    file_path = '/tmp/' + file.filename
    file.save(file_path)
    print(is_abs)
    

    # Process the file (e.g., perform some computation)
    # Example:
    # result = process_file_function(file_path)

    # Delete the temporary file
    # os.remove(file_path)
    output = process_input(file_path, smiles, 1200, is_abs)

    return jsonify({'result': output }), 200

@app.route('/predict', methods=['POST'])
def predict():

    return "Output PNG file path"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
