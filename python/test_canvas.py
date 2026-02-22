import tkinter as tk
import random

root = tk.Tk()
root.geometry("600x200")
canvas = tk.Canvas(root, bg="black", width=600, height=200)
canvas.pack()

data = [100.0] * 100

def update():
    data.append(random.uniform(0, 200))
    data.pop(0)
    canvas.delete("all")
    w, h = 600, 200
    step = w / len(data)
    points = []
    for j, v in enumerate(data):
        points.extend([j * step, h - v])
    canvas.create_line(points, fill="lime", width=2)
    root.after(50, update)

update()
root.mainloop()
