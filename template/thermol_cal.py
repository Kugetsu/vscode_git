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



#需要的输入信息包括DOS信息以及高频频率数值