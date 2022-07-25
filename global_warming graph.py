import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

df = pd.read_csv('C:/Users/Daniel/Downloads/GLB.Ts+dSST(1).csv', header=1)
x=df.Year[70:-1]
y=df.SON[70:-1]
x = pd.to_numeric(x)
y = pd.to_numeric(y)
plt.scatter(x, y, c=cm.hot(1-(y-y.min())/(y.max()-y.min())))
plt.axhline(1.5, color=cm.hot(1), linestyle='-', linewidth=5)
plt.axhline(0, color=cm.hot(1), linestyle=':')
plt.annotate("You are here",
            xy=(2022, 0.97), xycoords='data',
            xytext=(2020, 1.2), textcoords='data',
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="-[",
                            connectionstyle="angle3"),
            )
t = plt.text(
    2000, 1.55, "IRREVERSIBLE, RUNAWAY PROCESSES", ha="center", va="center", size=15, color='#330000')
plt.ylim(-0.25, 1.5)
plt.xlabel('Year')
plt.ylabel('Average temperature above norm (°C)')
plt.text(
    2004, -0.05, "Data: https://data.giss.nasa.gov/gistemp/", ha="center", va="center", size=8, color='black')
plt.show()