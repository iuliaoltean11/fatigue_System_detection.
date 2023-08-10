import re
import socket
import struct
from tkinter import *

import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Define initial values
x = [0.0]
x2 = [0.0]
y1 = [0.0]
y2 = [11]

class Start_interface():
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn.connect(('localhost', 1234))

   # conn2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  #  conn2.connect(('192.168.1.135', 1234))
    # Create the Tkinter window
    root = Tk()
    root.geometry("500x800")
    canvas = Canvas(root, width=400, height=200)
    canvas.pack()

    coord = 20, 50, 210, 210

    # Create the first Matplotlib figure and add the first line chart
    fig1 = plt.figure(figsize=(8, 2))
    ax1 = fig1.add_subplot(111)
    ax1.plot(x, y1, label='Grafic atentie')
    ax1.set_title('Attention data')
    ax1.set_xlabel('Nr. citiri')
    ax1.set_ylabel('Procentaj atentie')
    ax1.legend()

    # Create the second Matplotlib figure and add the second line chart
    fig2 = plt.figure(figsize=(8, 2))
    ax2 = fig2.add_subplot(111)
    ax2.plot(x, y2, label='Grafic nivel etanol')
    ax2.set_title('Ethanol data')
    ax2.set_xlabel('Nr. citiri')
    ax2.set_ylabel('Nivel etanol')
    ax2.legend()

    # Add the Matplotlib figures to Tkinter canvases
    canvas1 = FigureCanvasTkAgg(fig1, master=root)
    canvas1.draw()
    canvas1.get_tk_widget().pack()

    canvas2 = FigureCanvasTkAgg(fig2, master=root)
    canvas2.draw()
    canvas2.get_tk_widget().pack()

    def __init__(self):
        self.start()

    def start(self):
        # Start the Tkinter event loop
        self.root.after(1000, self.start_routine)
        self.root.mainloop()

    def add_value(self):
        # Redraw the charts
        self.ax1.plot(x, y1, label='Line Chart 1')
        self.ax2.plot(x2, y2, label='Line Chart 2')
        self.canvas1.draw()
        self.canvas2.draw()

        percentage = float("{:.2f}".format(y1[-1]))
        self.canvas.delete("all")
        self.canvas.create_text(120, 100, text=percentage, font=("Helvetica 30 bold"))
        self.arc = self.canvas.create_arc(self.coord, start=180, extent=-1 * (180 * percentage) / 100, style=ARC,
                                          width=30)

    def start_routine(self):
            self.populate()
            self.root.after(1000, self.start_routine)

    def populate(self):
        data = self.conn.recv(1024).decode()

        numbers = re.findall(r'\d+\.\d+', data)
        print(data)
        array = [float(num) for num in numbers]

        #self.ethanol_get()

        # print(array)
        for value in array:
            x.append(x[-1] + 1)
            y1.append(value*100)
        self.add_value()

    def ethanol_get(self):
        data2 = self.conn2.recv(4)
        numbers2 = struct.unpack('f', data2)[0]
        print(numbers2)
        # for value in numbers2:
        x2.append(x2[-1] + 1)
        y2.append(numbers2)
        print(y2)


if __name__ == '__main__':
    Start_interface()