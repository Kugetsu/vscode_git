import tkinter as tk
#主窗口 
root = tk.Tk()
root.title("ISAC_1.1.0")

#创建标签
label =tk.Label(root,text="hello!")
label.pack() #将标签添加到窗口中，自动调整位置

#按钮回调函数
def button_click():
    label.config(text="button")

#创建按钮
button = tk.Button(root, text="click",command=button_click)
button.pack()

#运行主循环
root.mainloop()