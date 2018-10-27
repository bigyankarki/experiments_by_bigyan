from flask import Flask, render_template, request
from api.digit_recognition.predict import return_prediction
from base64 import b64decode

app = Flask(__name__)

ht_for_app = {
    'apps' : ['digit_recognition', 'char_RNN'],
    'description' : ['digit recognition using ANN', 'Generate characters using RNN'],
    'bg' : ['digit.jpeg', 'charRNN.png']
}

@app.route("/")
def index():
    apps = ht_for_app['apps']
    description = ht_for_app['description']
    bg = ht_for_app['bg']

    return render_template('indexContent.html', apps=list(zip(apps,description,bg)))

@app.route("/<name>")
def apps(name):
    if name in ht_for_app['apps']:
        if name == 'digit_recognition':
            return render_template( name+'.html', name=name)
    else:
        return index()

@app.route("/api/<app_name>", methods=['GET', 'POST'])
def get_image(app_name):
    image_b64 = request.values['image']

    predict, confidence = return_prediction(image_b64)

    return [predict,confidence]

if __name__ == '__main__':
    app.run(debug=True)
