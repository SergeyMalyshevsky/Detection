import os

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from detection import detect_people

app = Flask(__name__)

UPLOAD_FOLDER = './static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def main():
    return render_template('./main.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('./show_image.html', image_file=filename)


@app.route('/detection', methods=['GET', 'POST'])
def detection():
    if request.method == 'POST':
        image_file = request.form['image_file']
        if image_file:
            filename = image_file
            detect_people(filename)
            return render_template('./result.html', image_file=filename)


if __name__ == '__main__':
    app.run(debug=True)
