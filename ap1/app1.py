from flask import Flask,request
from flask import render_template
import pandas as pd
import requests
import numpy as np
from preprocessing import Check
import pickle

app = Flask ( __name__ )
menu = ["Kiritishingiz kerak bo'lgan ustunlar:","Pclass -> [1:3] int","Name -> text string",'Sex -> male, female string','Age ->   [1:80] int','SibSp -> [0:5] int','Parch -> [0:6] int','Ticket -> price int','Fare -> price int','Embarked -> [S, C, Q] string']

@app.route ("/index")
@app.route ("/")
def index ():
    return render_template("index.html",title="Malumotlarni kiriting",menu=menu)

# @app.route ("/about")
# def about ( ) :
#     return "<h1> 0 сaйтe </h1>"

@app.route("/titanic",methods=["GET","POST"])
def profile():
    # info = username
    # check = Check(username)
    # new = check.Run()
    # print(new.T)
    # model = pickle.load(open("dtree_model.pkl","rb"))
    # pred = model.predict(new.T)


    info = request.get_data()
    info = str(info,'utf8')
    file = open('data.txt','w')
    file.write(info)
    file.close()



    data = pd.read_excel('/home/suxrob/Downloads/Untitled 1.ods', engine='odf')

    pred = []

    for w in range(len(data)):

        check = Check(list(data.iloc[w,:]))
        new = check.Run()
        model = pickle.load(open("dtree_model.pkl","rb"))
        pred.append(model.predict(new.T)[0])

    pred = pd.Series(pred)
    new = pd.concat([data,pred],axis=1)
    new.rename(columns = {0:"Predicts"},inplace=True)
    return  new.to_csv()

if __name__ == " __main__ " :
    app.run ( debug = True )

