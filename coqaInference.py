import os
import flask
from flask import request
from flask_cors import CORS
import json
from coqa_bert_inference import *

app = flask.Flask(__name__)
CORS(app)
BertModel = None

host = '0.0.0.0'
port = 9200
headers = {'Content-type': 'application/json'}


@app.route("/coqa_infer", methods=["POST"])
def getResponse():
    data = request.get_json()
    result = BertModel.getResult(data)
    output = {"extractive_answer":result[0][0]}
    return json.dumps(output)


if __name__ == "__main__":
    BertModel = ModelLoader(
        model_name='BERT', model_path='./output/output4000/best/model.pth', device='cuda', embedding_length = 35334)
    app.run(host=host, port=port, debug=True,
            threaded=True, use_reloader=False)
