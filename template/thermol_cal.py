#热力学计算部分单独编程，读取DOS信息
import os,sys,math,cmath
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d #鼠标点击交互
from tkinter import ttk
from tkinter import messagebox as mbox
from tkinter import filedialog as fd
import pandas as pd


kB=1.380649*10**(-23) #玻尔兹曼常数
h=6.62607015*10**(-34) #普朗克常量
L=6.022*10**23 #阿伏伽德罗常数
V=29979245800
e=math.e
base_num=6
T_range=4000

w=()
x=()
HF=()

file_name="DOS"
df_1=pd.read_excel(file_name,sheet_name='bs1')

#读取文件内的数据

plt.clf()
fig=plt.figure(figsize=(30,30))
ax = plt.axes(projection='2d')
ax.set_title('Cv')
for T in range(1,T_range):
    Cv_l=0
    for x in range(0,len(x)):
        K=h*(x+0.5)*V/(kB*T)
        C=L*kB*w(x)*K*K*e**(-K)/((e**(-K)-1)**2)
        Cv_l=Cv_l+C
    plt.plot(T,Cv_l,color='b',marker='x')

    Cv_h=0
    for i in range(0,len(HF)):
        K=h*HF[i]*V/(kB*T)
        C=L*kB*K*K*e**(-K)/((e**(-K)-1)**2)
        Cv_h=Cv_h+C
    plt.plot(T,Cv_h,color='r',marker='x')
    Cv_t=Cv_l+Cv_h
    plt.plot(T,Cv_t,color='black',mark='o')
    if (T==298):
        print("室温の熱容量 Cv=",Cv_t)
plt.show()



#需要的输入信息包括DOS信息以及高频频率数值