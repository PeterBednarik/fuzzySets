import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def display_fuzzy_set():  
    data2 = {'year': [0, 55, 90, 100],
         'NL': [1, 1, 0, 0]
         }  
    df2 = pd.DataFrame(data2)

    newWindow = tk.Tk()

    figure2 = plt.Figure(figsize=(5, 4), dpi=100)
    ax2 = figure2.add_subplot(111)
    line2 = FigureCanvasTkAgg(figure2, newWindow)
    line2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    df2 = df2[['year', 'NL']].groupby('year').sum()
    df2.plot(kind='line', legend=True, ax=ax2, color='r', fontsize=10)
    ax2.set_title('Fuzzy Set')


root= tk.Tk()

canvas1 = tk.Canvas(root, width=400, height=300)
canvas1.pack()

entry1 = tk.Entry(root) 
canvas1.create_window(200, 120, window=entry1)
entry2 = tk.Entry(root) 
canvas1.create_window(200, 160, window=entry2)
        
button1 = tk.Button(text='Display Fuzzy Set', command=display_fuzzy_set)
canvas1.create_window(200, 200, window=button1)

root.mainloop()
