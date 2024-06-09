# app.py
from flask import Flask, render_template 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import statistics

data = pd.read_csv("D:/MACHINE LEARNING/Indus Internship/web_plot2/Maintenance_Info2.csv")
data["Air temperature [K]"] = data["Air temperature [K]"] - 272.15
data["Process temperature [K]"] = data["Process temperature [K]"] - 272.15

data.rename(columns={"Air temperature [K]": "Air temperature [°C]",
                     "Process temperature [K]": "Process temperature [°C]"}, inplace=True)


data["Temperature difference [°C]"] = data["Process temperature [°C]"] - data["Air temperature [°C]"]

app = Flask('__main__')  # Changed from 'app' to '__main__'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plots')
def plots():

    # Generate plots
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

            # ax = plt.subplot(1, 2, 2)
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

if __name__ == '__main__':
    app.run(debug=True)