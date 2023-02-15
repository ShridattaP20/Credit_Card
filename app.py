import os
from flask import Flask, request, render_template, jsonify
import yaml
import joblib
import numpy as np
import flask
from prediction_service import prediction

 

webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)


@app.route("/", methods=["GET", "POST"])
def index():
    if flask.request.method == 'POST':
        try:
            if flask.request.form:
                data_req = dict(flask.request.form)
                response = prediction.form_response(data_req)
                return render_template('index.html', response=response)

            elif flask.request.json:
                response = prediction.api_response(flask.request.json)
                return jsonify(response)

        except Exception as e:
            print(e)
            error = {"error": e}
            return render_template("404.html", error=error)

    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
