# app.py
from flask import Flask, render_template 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

app = Flask('__main__')  # Changed from 'app' to '__main__'

@app.route('/')
def home():
    data = pd.read_csv("D:/MACHINE LEARNING/Indus Internship/web_plot2/Maintenance_Info2.csv")

    # Generate plots
    plots = ['histplot', 'barplot', 'scatterplot', 'boxplot', 'Pie chart']
    for plot in plots:
        if plot == 'histplot':
            sns.displot(data=data, x="Air temperature [K]", kde=True, bins = 100,color = "red", facecolor = "yellow",height = 5, aspect = 1.5);
        elif plot == 'barplot':
            sns.displot(data=data, x="Process temperature [K]", kde=True, bins = 100,color = "red", facecolor = "lime",height = 5, aspect = 1.5);
        elif plot == 'scatterplot':
            sns.scatterplot(data=data, x="Torque [Nm]", y="Rotational speed [rpm]", hue="Type",palette="bright");
        elif plot == 'boxplot':
            fig, ax = plt.subplots(2, 1, figsize=(5,8))
            sns.boxplot(data=data["Torque [Nm]"], ax = ax[0],color='grey')
        elif plot == 'Pie chart':
            ax = plt.figure(figsize=(20, 10))

            # ax = plt.subplot(1, 2, 1)
            # ax = sns.countplot(x='Type', data=data)
            # ax.bar_label(ax.containers[0])
            # plt.title("Type", fontsize=30, color='Red', font='Arial')

            ax = plt.subplot(1, 2, 1)
            ax = data['Type'].value_counts().plot.pie(explode=[0.1, 0.1, 0.1], autopct='%1.2f%%', shadow=True);
            ax.set_title(label="Type", fontsize=30, color='Red', font='Arial')

        plt.savefig(f'static/{plot}.png')
        plt.clf()



    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)