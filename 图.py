import matplotlib.pyplot as plt

from matplotlib import font_manager#导入字体管理模块
my_font = font_manager.FontProperties(fname="C:/WINDOWS/Fonts/STSONG.TTF")

x = [1,2,3,4,5,6,7,8,9,10]
y = [1.1665,0.7919,0.5057,0.4692,0.3732,0.3265,0.1870,0.3225,0.2787,0.1598]

plt.title("每轮训练损失/lr=0.001",fontproperties = my_font,fontsize = 18)    #设置标题
plt.xlabel("Epoch",fontproperties = my_font,fontsize = 18)    #设置x坐标标注，字体为18号
plt.ylabel("loss",fontproperties = my_font,fontsize = 18)    #设置y坐标标注



# 绘图
plt.plot(x, y)
# 显示
plt.show()