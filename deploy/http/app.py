from flask import Flask, jsonify,request
from PIL import Image
import io
app = Flask(__name__)


@app.route('/',methods=["POST"])
def rn():
    print(request.form.get('name'))
    file=request.files.get('file')
    print(file)
    file_type=file.read()
    img=Image.open(io.BytesIO(file_type))
    img.show()
    return jsonify({'class_id': 'IMAGE_NET_XXX', 'class_name': 'Cat'})


if __name__ == '__main__':
    app.run()

