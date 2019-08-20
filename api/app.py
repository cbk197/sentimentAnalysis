from flask import Flask, render_template,json, request,Response
from trainLSTM import Train
import os
import socket

# Connect to Redis

app = Flask(__name__)
mod = Train() 
@app.route("/")
def hello():
    # mod = Train()
    # if mod.predict_sentence("shop phuc vu kem, san pham khong duoc nhu mong doi :( ",'./data/word2vec.model2','./data/LSTM.model73') == 1:
    #   print("tieu cuc")
    # else:
    #   print("tic cuc")
    return  render_template('index.html')

@app.route("/welcome")
def welcome():
   
    return  json.dumps({'chuan':'van'})

@app.route("/postmt", methods=['POST'])
def post():
    # print( request.data.decode('utf-8'))
    text = request.data.decode('utf-8')
    # print(text)
    
    if mod.predict_sentence(text,'./data/word2vec.model2','./data/LSTM.model73') == 1:
        return  json.dumps({'result':u'tiêu cực'})
    else:
        return  json.dumps({'result':u'tích cực'})

@app.route("/getrantext")
def getText():
    text, label = mod.get_RanText('./data/train.crash')
    if label[0] == 1:
        lb = 0
    else:
        lb = 1
    return  json.dumps({'text':text , 'label' : lb})
    
if __name__ == "__main__":
    if os.environ.get('PORT') is not None:
        app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT'))
    else:
        app.run(debug=True, host='0.0.0.0') 