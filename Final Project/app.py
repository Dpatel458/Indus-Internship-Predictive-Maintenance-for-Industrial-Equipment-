# app.py
from flask import Flask, render_template,request
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statistics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from joblib import load,dump
import joblib

##-----------------------------------------------------------------------------------------------------------------

data = pd.read_csv("D:/MACHINE LEARNING/Indus Internship/web_plot2/Maintenance_Info2.csv")
data["Air temperature [K]"] = data["Air temperature [K]"] - 272.15
data["Process temperature [K]"] = data["Process temperature [K]"] - 272.15

data.rename(columns={"Air temperature [K]": "Air temperature [°C]",
                     "Process temperature [K]": "Process temperature [°C]"}, inplace=True)


data["Temperature difference [°C]"] = data["Process temperature [°C]"] - data["Air temperature [°C]"]


X, y = make_classification(n_samples=100, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)


clf = LogisticRegression(random_state=0).fit(X, y)


dump(clf, 'model.joblib') 


##-----------------------------------------------------------------------------------------------------------------


app = Flask(__name__)  


##-----------------------------------------------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

##-----------------------------------------------------------------------------------------------------------------

@app.route('/datadis')
def datadis():
    df=pd.read_csv("D:/MACHINE LEARNING/Indus Internship/web_plot2/Maintenance_Info2.csv")
    return df.to_html()

##----------------------------------------------------------------------------------------------------------------- 

@app.route('/plots')
def plots():

    
    plots = {'histplot', 'barplot', 'displot', 'plot4', 'plot5', 'plot6', 'plot7', 'plot8', 'scatterplot', 'plot9',
             'boxplot', 'plot10','plot11','plot12','plot13'}


    for plot in plots:
        if plot == 'histplot':
            sns.displot(data=data, x="Air temperature [°C]", kde=True, bins = 100,color = "red", facecolor = "yellow",height = 5, aspect = 1.5);
        elif plot == 'barplot':
            sns.displot(data=data, x="Process temperature [°C]", kde=True, bins = 100,color = "red", facecolor = "lime",height = 5, aspect = 1.5);
        elif plot == 'displot':
            sns.displot(data=data, x="Temperature difference [°C]", kde=True, bins=100, color="green",facecolor="orange", height=5, aspect=1.5);

        elif plot == 'scatterplot':
            plt.figure(figsize=(7.5, 5))
            sns.scatterplot(data=data, x="Torque [Nm]", y="Rotational speed [rpm]", hue="Type",palette="bright");

        elif plot == 'plot4':
            ax = plt.figure(figsize=(7.5,5))

            ax = plt.subplot(1, 2, 1)
            ax = sns.countplot(x='Type', data=data)
            ax.bar_label(ax.containers[0])
            plt.title("Type", fontsize=50, color='Red', font='Arial')

            ax = plt.subplot(1, 2, 2)
            ax = data['Type'].value_counts().plot.pie(explode=[0.1, 0.1, 0.1], autopct='%1.2f%%', shadow=True);
            ax.set_title(label="Type", fontsize=50, color='Red', font='Arial');

        elif plot == 'plot5':
            ax = plt.figure(figsize=(7.5,5))

            ax = plt.subplot(1, 2, 1)
            ax = sns.countplot(x='Target', data=data)
            ax.bar_label(ax.containers[0])
            plt.title("Target", fontsize=30, color='Red', font='Times New Roman')

            ax = plt.subplot(1, 2, 2)
            ax = data['Target'].value_counts().plot.pie(explode=[0.2, 0.2], autopct='%1.2f%%', shadow=True);
            ax.set_title(label="Target", fontsize=30, color='Red', font='Times New Roman');

        elif plot == 'plot6':
            ax = plt.figure(figsize=(7.5,5))

            
            ax = data['Failure Type'].value_counts().plot.pie();
            ax.set_title(label="Target", fontsize=30, color='Red', font='Times New Roman');

        elif plot =='plot7':
            plt.figure(figsize=(7.5,5))
            sns.scatterplot(data=data, x="Torque [Nm]", y="Rotational speed [rpm]", hue="Failure Type", palette="deep");


        elif plot == 'plot8':
            plt.figure(figsize=(7.5,5))
            sns.scatterplot(data=data, x="Torque [Nm]", y="Rotational speed [rpm]", hue="Target", palette="dark");

        elif plot == 'boxplot':
            fig, ax = plt.subplots( figsize=(7.5,5))
            sns.boxplot(data=data["Torque [Nm]"],color='cyan')

        elif plot == 'plot9':
            feature = "Torque [Nm]"

            fig, ax = plt.subplots(figsize=(7.5,5))

            sns.histplot(data=data[feature], kde=True, color='yellow', facecolor="cyan", ax=ax)

            ax.axvline(x=data[feature].mean(), color='Magenta', linestyle='--', linewidth=4,label='Mean: {}'.format(round(data[feature].mean(), 3)))
            ax.axvline(x=data[feature].median(), color='lime', linewidth=3,label='Median: {}'.format(round(data[feature].median(), 3)))
            ax.axvline(x=statistics.mode(data[feature]), color='brown', linewidth=2,label='Mode: {}'.format(statistics.mode(data[feature])))
            ax.legend()


        elif plot == 'plot10':
            feature = "Rotational speed [rpm]"

            fig, ax = plt.subplots(figsize=(7.5,5))

            sns.histplot(data=data[feature], kde=True, color='yellow', facecolor="green", ax=ax)

            ax.axvline(x=data[feature].mean(), color='Magenta', linestyle='--', linewidth=4,label='Mean: {}'.format(round(data[feature].mean(), 3)))
            ax.axvline(x=data[feature].median(), color='lime', linewidth=3,label='Median: {}'.format(round(data[feature].median(), 3)))
            ax.axvline(x=statistics.mode(data[feature]), color='brown', linewidth=2,label='Mode: {}'.format(statistics.mode(data[feature])))
            ax.legend()

        elif plot == 'plot11':
            fig, ax = plt.subplots( figsize=(7.5,5))
            sns.boxplot(data=data["Rotational speed [rpm]"],color='green')

    

        elif plot == 'plot12':
            sns.displot(data=data, x="Torque [Nm]", col="Type", kind="kde")



        plt.savefig(f'static/{plot}.png')
        plt.clf()



    return render_template('plots.html')



##-----------------------------------------------------------------------------------------------------------------


@app.route('/model')
def model():
    import matplotlib.pyplot as plt

    
    accuracies = {
        'Logistic Regression': 96.75,
        'Decision Tree': 92.00,
        'Random Forest': 96.75,
        'SVM': 96.75
    }

    
    models = list(accuracies.keys())
    values = list(accuracies.values())

    
    colors = ['blue', 'green', 'red', 'orange']

    
    plt.figure(figsize=(10, 5))
    bars=sns.barplot(x=models, y=values, palette=colors)



    for bar, value in zip(bars.patches, values):
        plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height(), 
             f'{value}%', 
             ha='center', 
             va='bottom')


    
    plt.xlabel('Models')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracies')
    plt.ylim(0, 100)

    plt.savefig('static/model_accuracies.png')


    return render_template('model.html', model_image='model_accuracies.png')

le = LabelEncoder()
x = data.drop(columns="Failure Type").apply(le.fit_transform)
y = data["Failure Type"]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=21)

log = LogisticRegression()
log.fit(x_train, y_train)

y_pred = log.predict(x_test)


##----------------------------------------------------------------------------------------------------------------------


@app.route('/lr')
def lr():
    
    a = accuracy_score(y_test, y_pred) * 100
    b = precision_score(y_test, y_pred, average='macro')
    c = recall_score(y_test, y_pred, average='macro')
    d = f1_score(y_test, y_pred, average='macro')
    

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    scores = [a, b, c, d]

    
    plt.figure(figsize=(8, 6))
    bar1=sns.barplot(x=metrics, y=scores, palette='viridis')

    for i, score in enumerate(scores):
        bar1.text(i, score, round(score, 2), ha = 'center', va = 'bottom')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Metrics', fontsize=20)
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.savefig('static/LR_data.png')

    plt.close()

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='crest', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix', fontsize=20)
    plt.savefig('static/LR_confusion_matrix.png')
    plt.close()

    return render_template('lr.html', model_image='LR_data.png', cm_image='LR_confusion_matrix.png')


##-----------------------------------------------------------------------------------------------------------------


@app.route('/', methods=['GET'])
def input_form():
    return render_template('input.html')

model = load('model.joblib')

df = pd.read_csv('Maintenance_Info2.csv')
features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]']
X = df[features]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

joblib.dump(model, 'logistic_regression_model.pkl')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        
        model = joblib.load('logistic_regression_model.pkl')

        features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]']
        data = {}
        for feature in features:
            value = float(request.form[feature])  # Convert string to float
            data[feature] = value

        input_df = pd.DataFrame(data, index=[0])
        prediction = model.predict(input_df)
        return render_template('predict.html', prediction=prediction)

    elif request.method == 'GET':
        return render_template('input.html')




##----------------------------------------------------------------------------------------------------------------- 

if __name__ == '__main__':
    app.run(debug=True)