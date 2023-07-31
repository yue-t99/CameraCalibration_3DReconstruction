from numpy import *
import numpy as np
import sys
import re
import cv2 as cv
from numpy.linalg import  *
from skimage import data,draw,color,transform,feature
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import digital

font = cv.FONT_HERSHEY_SIMPLEX

width=3872
height=2592
x0=1935.5000000000 
y0=1295.5000000000
f=7935.786962 
def separate_color(Pointimg):
    # cv.namedWindow("image",cv.WINDOW_NORMAL) 
    # cv.imshow("image", Pointimg)
    # cv.setMouseCallback("image", getposBgr)
    hsv = cv.cvtColor(Pointimg, cv.COLOR_BGR2HSV)                              #色彩空间转换为hsv，便于分离
    lower_hsv = np.array([0, 0, 0])                                      #提取颜色的低值
    high_hsv = np.array([30, 30, 30])                                     #提取颜色的高值
    mask = cv.inRange(hsv, lowerb = lower_hsv, upperb = high_hsv) 
    # cv.imshow("inRange", mask) 
    # cv.waitKey(0)        
    return mask 
# 分离出图中的黑色像素

def getposBgr(event, x, y, flags, param):
    if event==cv.EVENT_LBUTTONDOWN:
        print("Bgr is", Pointimg[y, x])
# 鼠标点击图像区域，可以打印出该点的RGB，便于阈值的设定

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y) 
        print('x, y = {}, {}'.format(x, y)) 

def Pointremain(img):
    cv.namedWindow("image",cv.WINDOW_NORMAL) 
    cv.imshow("image", img) 
    
    loc = cv.setMouseCallback("image", on_EVENT_LBUTTONDOWN) 
    cv.waitKey(0) 
# 鼠标点击图像区域，可以打印出该点的xy，便于阈值的设定

def xy2rstheta(x,center):
    r=math.sqrt(math.pow(x[0]-center[0],2)+math.pow(x[1]-center[1],2)) 
    theta=math.atan2(x[1]-center[1],x[0]-center[0])/math.pi*180 
    if(theta<0):
        theta=360+theta 
    return [r,theta]
# 极坐标转换

def GetPointnum(img):
    h, w = img.shape[:2]      #获取图像的高和宽 (579,509)

    C_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)  #把输入图像灰度化
    C_binary =  cv.adaptiveThreshold(C_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, 25, 10)
    C_re,h = cv.findContours(C_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 
    for c in C_re:
        if len(c)>50:
            S1= cv.contourArea(c) 
            ell=cv.fitEllipse(c)
            S2 =math.pi*ell[1][0]*ell[1][1]
            if (S1/S2)>0.23 :#根据面积比例 判断是否为圆（椭圆）
                CenterPoint=ell[0] 
    # 获取中心像素

    Pointimg=separate_color(img) 
    # 从原图中分离出黑色像素 获取控制点的信息 同时完成二值化
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5, 5)) 
    dilation = cv.dilate(Pointimg,kernel,iterations = 1)
    # 像素膨胀
    contours, hier = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 
    # findContours函数查找图像里的图形轮廓           
    black = cv.cvtColor(np.zeros((Pointimg.shape[0], Pointimg.shape[1]), dtype=np.uint8), cv.COLOR_GRAY2BGR) 
    # 创建新的图像black
    
    P_ell_temp=[] 
    P_ell_xy=[]
    for cnt in contours:
        if len(cnt)>5:
            S1= cv.contourArea(cnt) 
            ell=cv.fitEllipse(cnt)
            S2 =math.pi*ell[1][0]*ell[1][1] 
            ob=[] 
            if (S1/S2)>0.1 :#根据面积比例 判断是否为圆（椭圆）
                black = cv.ellipse(black, ell, (0, 255, 0), 2) 
                black = cv.circle(black, (int(ell[0][0]),int(ell[0][1])), 1, (0,0,255), 4)
                rtheta_temp=xy2rstheta(ell[0],CenterPoint) 
                ob.append(ell) 
                ob.append(rtheta_temp[0]) 
                ob.append(rtheta_temp[1]) 
                P_ell_temp.append(ob) 
    # 获取各个小圆 及其 极坐标
    P_ell_temp.sort(key=lambda P_ell_temp: P_ell_temp[2]) 
    i=0 
    Point_num=np.zeros(60) 
    while (i<60):
        ell_temp=P_ell_temp[i:i+4] 
        ell_temp.sort(key=lambda ell_temp: ell_temp[1],reverse=False) 
        for n in range(0,4):
            if(ell_temp[n][0][1][1]>13.5):
                Point_num[60-i-n-1]=1  
            black=cv.putText(black, str(60-i-n-1), (int(ell_temp[n][0][0][0]), int(ell_temp[n][0][0][1])), font, 0.6, (255, 255, 0), 1)      
        i=i+4 
    # 根据极坐标确定点编号
    # cv.imshow("test",black) 
    # cv.waitKey(0)   
    return Point_num 
# 获取编号规则

def ControlPointdetect(img,Point_num,P_coor):
    h, w = img.shape[:2]      #获取图像的高和宽 (579,509)
    C_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)  #把输入图像灰度化
    m = np.reshape(C_gray, [1,w*h])
    mean = m.sum()/(w*h)
    ret, C_binary =  cv.threshold(C_gray, mean, 255, cv.THRESH_BINARY) 
    C_Canny=cv.Canny(C_binary,200,300) 
    C_re,h = cv.findContours(C_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 
    black = cv.cvtColor(np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8), cv.COLOR_GRAY2BGR) 
    line=[] 
    for c in C_re:
        if len(c)>50:
            S1= cv.contourArea(c) 
            ell=cv.fitEllipse(c) 
            S2 =math.pi*ell[1][0]*ell[1][1] 
            if (S1/S2)>0.05 :#根据面积比例 判断是否为圆（椭圆）
                line=c 
    C_re=[] 
    for temp in line:
        if(temp[0,1]>1200):
            # 1000 是根据鼠标点击检测得到
            C_re.append(temp) 
    C_re=np.array(C_re) 
    ell=cv.fitEllipse(C_re) 
    cv.ellipse(img, ell, (0, 255, 0), 5) 
    CenterPoint=ell[0] 
    # 获取中心像素

    contours, hier = cv.findContours(C_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 
    # findContours函数查找图像里的图形轮廓           
    P_ell_temp=[] 
    for cnt in contours:
        if len(cnt)>50:
            S1= cv.contourArea(cnt) 
            ell=cv.fitEllipse(cnt)
            S2 =math.pi*ell[1][0]*ell[1][1] 
            ob=[] 
            if (S1/S2)>0.24 :#根据面积比例 判断是否为圆（椭圆）
                # black = cv.ellipse(img, ell, (0, 255, 0), 5) 
                # black = cv.circle(black, (int(ell[0][0]),int(ell[0][1])), 1, (0,0,255), 4)
                rtheta_temp=xy2rstheta(ell[0],CenterPoint) 
                ob.append(ell) 
                ob.append(rtheta_temp[0]) 
                ob.append(rtheta_temp[1]) 
                P_ell_temp.append(ob) 
    # 获取各个小圆 及其 极坐标
    P_ell_temp.sort(key=lambda P_ell_temp: P_ell_temp[2]) 
    # 根据距离中心的极坐标排序
    ell_num=len(P_ell_temp)
    black = cv.circle(img, (int(CenterPoint[0]),int(CenterPoint[1])), 5, (0,0,255), 4) 
    i=0
    P_ell=[]
    ell_temp=[]
    while (i<ell_num):
        k=1
        ell_temp=[]
        ell_temp.append(P_ell_temp[i])
        # black = cv.ellipse(img, P_ell_temp[i][0], ( i*3, 255-i*3 , 0), 5)
        if P_ell_temp[i][0][1][1]>120:
            num=1
            ell_temp[0].append(num)
        else:
            num=0
            ell_temp[0].append(num)
        black=cv.putText(img, str(num), (int(P_ell_temp[i][0][0][0]), int(P_ell_temp[i][0][0][1])), font, 1.6, (255, 255, 0), 8)
        while (i+k<ell_num and P_ell_temp[i+k][2]-ell_temp[0][2]<7.5):
            ell_temp.append(P_ell_temp[i+k])
            # black = cv.ellipse(img, P_ell_temp[i+k][0], ( i*3, 255-i*3 , 0), 5)
            if P_ell_temp[i+k][0][1][1]>120:
                num=1
                ell_temp[k].append(num)
            else:
                num=0
                ell_temp[k].append(num)
            black=cv.putText(img, str(num), (int(P_ell_temp[i+k][0][0][0]), int(P_ell_temp[i+k][0][0][1])), font, 1.6, (255, 255, 0), 8)
            k=k+1 
        ell_temp.sort(key=lambda ell_temp: ell_temp[1], reverse=True) 
        P_ell.append(ell_temp)
        # cv.namedWindow("test",cv.WINDOW_NORMAL) 
        # cv.imshow("test",black) 
        # cv.waitKey(0)         
        i=i+k  
    P_num=len(P_ell)
    i=0

    while i<P_num :
        k=0
        if len(P_ell[i])==4:
            while k<len(Point_num) and len(P_ell[i][0])==4:
                if P_ell[i][0][3]==Point_num[k] and P_ell[i][1][3]==Point_num[k+1] \
                    and P_ell[i][2][3]==Point_num[k+2] and P_ell[i][3][3]==Point_num[k+3] :
                    P_ell[i][0].append(k)
                    P_ell[i][1].append(k+1)
                    P_ell[i][2].append(k+2)
                    P_ell[i][3].append(k+3)   
                    P_ell[i][0].append(P_coor[k])
                    P_ell[i][1].append(P_coor[k+1])
                    P_ell[i][2].append(P_coor[k+2])
                    P_ell[i][3].append(P_coor[k+3])
                    break 
                k=k+4
            for ell in P_ell[i]:
                black=cv.putText(img, str(ell[4]), (int(ell[0][0][0]), int(ell[0][0][1])), font, 1.6, (255, 255, 0), 8) 
            i=i+1
        else:
            if len(P_ell[i-1])==4 and len(P_ell[i-1][0])>4:
                if(P_ell[i-1][0][4]>3):
                    P_ell[i][0].append(P_ell[i-1][0][4]-4)
                    P_ell[i][1].append(P_ell[i-1][1][4]-4)
                    P_ell[i][2].append(P_ell[i-1][2][4]-4)
                    P_ell[i][0].append(P_coor[int(P_ell[i-1][0][4]-4)])
                    P_ell[i][1].append(P_coor[int(P_ell[i-1][1][4]-4)])
                    P_ell[i][2].append(P_coor[int(P_ell[i-1][2][4]-4)])
                else:
                    P_ell[i][0].append(P_ell[i-1][0][4]-4+60)
                    P_ell[i][1].append(P_ell[i-1][1][4]-4+60)
                    P_ell[i][2].append(P_ell[i-1][2][4]-4+60)
                    P_ell[i][0].append(P_coor[int(P_ell[i-1][0][4]-4+60)])
                    P_ell[i][1].append(P_coor[int(P_ell[i-1][1][4]-4+60)])
                    P_ell[i][2].append(P_coor[int(P_ell[i-1][2][4]-4+60)])
            else: 
                while k<len(Point_num) and len(P_ell[i+1][0])==4:
                    if P_ell[i+1][0][3]==Point_num[k] and P_ell[i+1][1][3]==Point_num[k+1] \
                        and P_ell[i+1][2][3]==Point_num[k+2] and P_ell[i+1][3][3]==Point_num[k+3] :
                        P_ell[i+1][0].append(k)
                        P_ell[i+1][1].append(k+1)
                        P_ell[i+1][2].append(k+2)
                        P_ell[i+1][3].append(k+3)
                        P_ell[i+1][0].append(P_coor[k])
                        P_ell[i+1][1].append(P_coor[k+1])
                        P_ell[i+1][2].append(P_coor[k+2])
                        P_ell[i+1][3].append(P_coor[k+3])
                        break 
                    k=k+4
                if(P_ell[i+1][0][4]<55):
                    P_ell[i][0].append(P_ell[i+1][0][4]+4)
                    P_ell[i][1].append(P_ell[i+1][1][4]+4)
                    P_ell[i][2].append(P_ell[i+1][2][4]+4)
                    P_ell[i][0].append(P_coor[int(P_ell[i+1][0][4]+4)])
                    P_ell[i][1].append(P_coor[int(P_ell[i+1][1][4]+4)])
                    P_ell[i][2].append(P_coor[int(P_ell[i+1][2][4]+4)])
                else:
                    P_ell[i][0].append(P_ell[i+1][0][4]+4-60)
                    P_ell[i][1].append(P_ell[i+1][1][4]+4-60)
                    P_ell[i][2].append(P_ell[i+1][2][4]+4-60)
                    P_ell[i][0].append(P_coor[int(P_ell[i+1][0][4]+4-60)])
                    P_ell[i][1].append(P_coor[int(P_ell[i+1][1][4]+4-60)])
                    P_ell[i][2].append(P_coor[int(P_ell[i+1][2][4]+4-60)])
            for ell in P_ell[i]:
                black=cv.putText(img, str(ell[4]), (int(ell[0][0][0]), int(ell[0][0][1])), font, 1.6, (255, 255, 0), 8) 
            i=i+1  
    # cv.namedWindow("test",cv.WINDOW_NORMAL) 
    # cv.imshow("test",black) 
    # cv.waitKey(0)         
    # 给每个小椭圆编号
    return P_ell    
# 获取图片上的控制点

def readTxt(filePath):
    result = []
    with open(filePath, 'r') as f:
        for line in f:
            temp = re.findall(r'\d+\s?\d+.\d+\s?\d+.\d+\s?\d+.\d+\s?\d\s?', line)
            arrPair = line.split( )
            if(temp):
                coor=[]
                for i in arrPair:
                    coor.append(float(i))
                result.append(coor)
    return result
# 读取控制点坐标

def outputxyandXY_3(P_ell,img):
    i=0
    num=len(P_ell)
    P_temp=[]
    xy=[]
    XY_3=[]
    while i<num:
        for ell in P_ell[i]:
            P_temp.append(ell)
        i=i+1
    P_temp.sort(key=lambda P_temp: P_temp[4]) 
    i=0
    while i<len(P_temp):
        temp=[P_temp[i][0][0][0],height-P_temp[i][0][0][1]]
        xy.append(temp)
        XY_3.append(P_temp[i][5])
        i=i+1
    return xy,XY_3,P_temp
# 整理控制点数据

def DLTcalcu(xy,XY_3,img):
    num=len(XY_3)
    i=0
    A=[]
    Z=[]
    while(i<num):
        s1=[XY_3[i][0],XY_3[i][1],1,0,0,0,-xy[i][0]*XY_3[i][0],-xy[i][0]*XY_3[i][1]]
        s2=[0,0,0,XY_3[i][0],XY_3[i][1],1,-xy[i][1]*XY_3[i][0],-xy[i][1]*XY_3[i][1]]
        A.append(s1)
        A.append(s2)
        Z.append(xy[i][0])
        Z.append(xy[i][1])
        i=i+1
    A=np.array(A)
    # print(A)
    h=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(A),A)),np.transpose(A)),Z)
    # h=np.linalg.lstsq(A,Z)
    # h=h[0][:]
    i=0
    cancha=np.zeros(num*2)
    while i<num:
        x=(h[0]*XY_3[i][0]+h[1]*XY_3[i][1]+h[2])/(h[6]*XY_3[i][0]+h[7]*XY_3[i][1]+1)
        y=(h[3]*XY_3[i][0]+h[4]*XY_3[i][1]+h[5])/(h[6]*XY_3[i][0]+h[7]*XY_3[i][1]+1)
        cancha[i*2]=xy[i][0]-x
        cancha[i*2+1]=xy[i][1]-y
        i=i+1
    # print(cancha)
    return h
# 计算DLT参数

def Externalele(h,x0,y0,f):
    kapa=math.atan((h[1]-h[7]*x0)/(h[4]-h[7]*y0))+math.pi#*180/math.pi
    b3=1/(1+(h[1]-h[7]*x0)**2/(f**2*h[7]**2)+(h[4]-h[7]*y0)**2/(f**2*h[7]**2))
    b3=sqrt(b3)
    b1=b3*(-(h[1]-h[7]*x0)/(f*h[7]))
    b2=b3*(-(h[4]-h[7]*y0)/(f*h[7]))
    if(abs(tan(kapa)-b1/b2)>0.1):
        b3=-b3
    omega=math.asin(-b3)#*180/math.pi
    b1=b3*(-(h[1]-h[7]*x0)/(f*h[7]))
    b2=b3*(-(h[4]-h[7]*y0)/(f*h[7]))
    temp1=-(h[0]-h[6]*x0)/(f*h[6])
    temp2=-(h[3]-h[6]*y0)/(f*h[6])
    phi=math.atan(-1/(temp1*b2-temp2*b1))#*180/math.pi
    R=calcuR(phi,omega,kapa) 
    lamb=np.zeros(6)
    lamb[0]=R[0]/((h[0]-h[6]*x0)/f)
    lamb[1]=R[3]/((h[1]-h[7]*x0)/f)
    lamb[2]=R[1]/((h[3]-h[6]*y0)/f)
    lamb[3]=R[4]/((h[4]-h[7]*y0)/f)
    lamb[4]=R[2]/(-h[6])
    lamb[5]=R[5]/(-h[7])
    lambda_ele=np.median(lamb)
    b=[(h[2]-x0)*(-lambda_ele/f),(h[5]-y0)*(-lambda_ele/f),lambda_ele]
    A=np.reshape(R,(3,3))
    A=np.transpose(A)
    re_XYZs=np.dot(np.linalg.inv(A),b)
    # kapa=kapa*180/math.pi
    # omega=omega*180/math.pi
    # phi=phi*180/math.pi
    return omega,kapa,phi,re_XYZs
# 计算外方位元素初值

def calcuR(phi,omega,kapa):
    R=np.zeros(9)
    R[0]=cos(phi)*cos(kapa)-sin(phi)*sin(omega)*sin(kapa)
    R[1]=-cos(phi)*sin(kapa)-sin(phi)*sin(omega)*cos(kapa)
    R[2]=-sin(phi)*cos(omega)
    R[3]=cos(omega)*sin(kapa)
    R[4]=cos(omega)*cos(kapa)
    R[5]=-sin(omega)
    R[6]=sin(phi)*cos(kapa)+cos(phi)*sin(omega)*sin(kapa)
    R[7]=-sin(phi)*sin(kapa)+cos(phi)*sin(omega)*cos(kapa)
    R[8]=cos(phi)*cos(omega)   
    return R
# 计算旋转矩阵

def calcuA(R,xy,XY_3,f,x0,y0,fin):
    i=0
    num=len(XY_3)
    temp=np.zeros(3*num)
    A=np.zeros(6*num*2)
    xy_temp=np.reshape(xy,-1)
    XYZ_temp=np.reshape(XY_3,-1)
    # re_A=[]
    while(i<num):
        temp[i * 3 + 0] = R[0] * (XYZ_temp[i * 3 + 0] - fin[0]) + R[3] * (XYZ_temp[i * 3 + 1] - fin[1]) + R[6] * (XYZ_temp[i * 3 + 2] - fin[2])
        temp[i * 3 + 1] = R[1] * (XYZ_temp[i * 3 + 0] - fin[0]) + R[4] * (XYZ_temp[i * 3 + 1] - fin[1]) + R[7] * (XYZ_temp[i * 3 + 2] - fin[2])
        temp[i * 3 + 2] = R[2] * (XYZ_temp[i * 3 + 0] - fin[0]) + R[5] * (XYZ_temp[i * 3 + 1] - fin[1]) + R[8] * (XYZ_temp[i * 3 + 2] - fin[2])
        A[i * 12 + 0] = (R[0] * f + R[2] * (xy_temp[i * 2]-x0)) / temp[i * 3 + 2]
        A[i * 12 + 1] = (R[3] * f + R[5] * (xy_temp[i * 2]-x0)) / temp[i * 3 + 2]
        A[i * 12 + 2] = (R[6] * f + R[8] * (xy_temp[i * 2]-x0)) / temp[i * 3 + 2]
        A[i * 12 + 3] = (xy_temp[i * 2 + 1]-y0) * sin(fin[4]) - ((xy_temp[i * 2]-x0) / f * ((xy_temp[i * 2]-x0) * cos(fin[5]) - (xy_temp[i * 2 + 1]-y0) * sin(fin[5])) + f * cos(fin[5]))*cos(fin[4])
        A[i * 12 + 4] = -f * sin(fin[5]) - (xy_temp[i * 2]-x0) / f * ((xy_temp[i * 2]-x0) * sin(fin[5]) + (xy_temp[i * 2 + 1]-y0) * cos(fin[5]))
        A[i * 12 + 5] = (xy_temp[i * 2 + 1]-y0)
        A[i * 12 + 6] = (R[1] * f + R[2] * (xy_temp[i * 2 + 1]-y0)) / temp[i * 3 + 2]
        A[i * 12 + 7] = (R[4] * f + R[5] * (xy_temp[i * 2 + 1]-y0)) / temp[i * 3 + 2]
        A[i * 12 + 8] = (R[7] * f + R[8] * (xy_temp[i * 2 + 1]-y0)) / temp[i * 3 + 2]
        A[i * 12 + 9] = -(xy_temp[i * 2]-x0) * sin(fin[4]) - ((xy_temp[i * 2 + 1]-y0) / f * ((xy_temp[i * 2]-x0) * cos(fin[5]) - (xy_temp[i * 2 + 1]-y0) * sin(fin[5])) - f * sin(fin[5]))*cos(fin[4])
        A[i * 12 + 10] = -f * cos(fin[5]) - (xy_temp[i * 2 + 1]-y0) / f * ((xy_temp[i * 2]-x0) * sin(fin[5]) + (xy_temp[i * 2 + 1]-y0) * cos(fin[5]))
        A[i * 12 + 11] = -(xy_temp[i * 2]-x0)  
        # re_A.append(A)
        i=i+1
    A=np.reshape(A,(-1,6))
    return A
# 计算基于共线方程的误差方程的系数阵

def calcuxy(R,fin,XY_3,f,x0,y0):
    i=0
    num=len(XY_3)
    XYZ_temp=np.reshape(XY_3,-1)
    re_xy=np.zeros(2*num)
    while i<num:
        re_xy[i * 2] = -f * (R[0] * ( XYZ_temp[i * 3] - fin[0]) + R[3] * ( XYZ_temp[i * 3 + 1] - fin[1]) + R[6] * (XYZ_temp[i * 3 + 2] - fin[2])) /(R[2] * (XYZ_temp[i * 3] - fin[0]) + R[5] * (XYZ_temp[i * 3 + 1] - fin[1]) + R[8] * (XYZ_temp[i * 3 + 2] - fin[2]))+x0
        re_xy[i * 2 + 1] = -f * (R[1] * (XYZ_temp[i * 3] - fin[0]) + R[4] * (XYZ_temp[i * 3 + 1] - fin[1]) + R[7] * (XYZ_temp[i * 3 + 2] - fin[2])) /(R[2] * (XYZ_temp[i * 3] - fin[0]) + R[5] * (XYZ_temp[i * 3 + 1] - fin[1]) + R[8] * (XYZ_temp[i * 3 + 2] - fin[2])) +y0
        i=i+1
    re_xy=np.reshape(re_xy,-1)
    return  re_xy
# 计算像点坐标

def E_ele_calcu(omega,kapa,phi,re_XYZs,xy,XY_3,f,x0,y0):
    i=0
    num=len(XY_3)
    R=calcuR(phi,omega,kapa)
    fin=[re_XYZs[0],re_XYZs[1],re_XYZs[2],phi,omega,kapa]
    A=calcuA(R,xy,XY_3,f,x0,y0,fin)
    re_xy=calcuxy(R,fin,XY_3,f,x0,y0)
    xy_temp=np.reshape(xy,-1)
    L=xy_temp-re_xy
    delta=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(A),A)),np.transpose(A)),L)
    p = (0.0314 / 180.0) / (60*60)
    while(abs(delta[3])>p or abs(delta[4])>p or abs(delta[5])>p):
        fin=fin+delta
        R=calcuR(fin[3],fin[4],fin[5])
        A=calcuA(R,xy,XY_3,f,x0,y0,fin)
        re_xy=calcuxy(R,fin,XY_3,f,x0,y0)
        L=xy_temp-re_xy
        delta=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(A),A)),np.transpose(A)),L)
    return fin
# 迭代解算外方位元素精确值

def thesamepoint(P_ell_l,P_ell_r):
    i=0
    num=len(P_ell_l)
    xy_r=[]
    xy_l=[]
    XYZ=[]
    while i<num:
        temp_r=[]
        temp_l=[]
        for ell in P_ell_r:
            if (P_ell_l[i][4]==ell[4]):
                temp_l=[P_ell_l[i][0][0][0],height-P_ell_l[i][0][0][1]]
                temp_r=[ell[0][0][0],height-ell[0][0][1]]
                xy_l.append(temp_l)
                xy_r.append(temp_r)
                XYZ.append(P_ell_l[i][5])
                break
        i=i+1
    return xy_l,xy_r,XYZ
# 获取两张影像的同名控制点

def PointProjectionele(fin_r,fin_l,xy_r,xy_l,XYZ,f):
    i=0
    num=len(xy_l)
    R_l=calcuR(fin_l[3],fin_l[4],fin_l[5])
    R_r=calcuR(fin_r[3],fin_r[4],fin_r[5])
    res=[]
    while i<num:
        A=[]
        b=[]
        l1=f*R_l[0]+(xy_l[i][0]-x0)*R_l[2]
        l2=f*R_l[3]+(xy_l[i][0]-x0)*R_l[5]
        l3=f*R_l[6]+(xy_l[i][0]-x0)*R_l[8]
        lx=f*R_l[0]*fin_l[0]+f*R_l[3]*fin_l[1]+f*R_l[6]*fin_l[2]+(xy_l[i][0]-x0)*R_l[2]*fin_l[0]+(xy_l[i][0]-x0)*R_l[5]*fin_l[1]+(xy_l[i][0]-x0)*R_l[8]*fin_l[2]
        l4=f*R_l[1]+(xy_l[i][1]-y0)*R_l[2]
        l5=f*R_l[4]+(xy_l[i][1]-y0)*R_l[5]
        l6=f*R_l[7]+(xy_l[i][1]-y0)*R_l[8]
        ly=f*R_l[1]*fin_l[0]+f*R_l[4]*fin_l[1]+f*R_l[7]*fin_l[2]+(xy_l[i][1]-y0)*R_l[2]*fin_l[0]+(xy_l[i][1]-y0)*R_l[5]*fin_l[1]+(xy_l[i][1]-y0)*R_l[8]*fin_l[2]
        s1=[l1,l2,l3]
        s2=[l4,l5,l6]
        A.append(s1)
        A.append(s2)
        b.append(lx)
        b.append(ly)
        l1=f*R_r[0]+(xy_r[i][0]-x0)*R_r[2]
        l2=f*R_r[3]+(xy_r[i][0]-x0)*R_r[5]
        l3=f*R_r[6]+(xy_r[i][0]-x0)*R_r[8]
        lx=f*R_r[0]*fin_r[0]+f*R_r[3]*fin_r[1]+f*R_r[6]*fin_r[2]+(xy_r[i][0]-x0)*R_r[2]*fin_r[0]+(xy_r[i][0]-x0)*R_r[5]*fin_r[1]+(xy_r[i][0]-x0)*R_r[8]*fin_r[2]
        l4=f*R_r[1]+(xy_r[i][1]-y0)*R_r[2]
        l5=f*R_r[4]+(xy_r[i][1]-y0)*R_r[5]
        l6=f*R_r[7]+(xy_r[i][1]-y0)*R_r[8]
        ly=f*R_r[1]*fin_r[0]+f*R_r[4]*fin_r[1]+f*R_r[7]*fin_r[2]+(xy_r[i][1]-y0)*R_r[2]*fin_r[0]+(xy_r[i][1]-y0)*R_r[5]*fin_r[1]+(xy_r[i][1]-y0)*R_r[8]*fin_r[2]
        s1=[l1,l2,l3]
        s2=[l4,l5,l6]
        A.append(s1)
        A.append(s2)
        b.append(lx)
        b.append(ly)
        res_temp=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(A),A)),np.transpose(A)),b)-XYZ[i]
        res.append(res_temp)
        i=i+1
    res=np.reshape(res,(-1,3))
    res_X=np.mean(res[:,0])
    res_Y=np.mean(res[:,1])
    res_Z=np.mean(res[:,2])
    print(res_X)
    print(res_Y)
    print(res_Z)
    return res
# 控制点三维坐标解算 并输出精度

def SIFT(img1,img2):
    MIN_MATCH_COUNT=10
    # Initiate SIFT detector
    orb = cv.ORB_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True) # 匹配描述符.
    matches = bf.match(des1,des2) # 根据距离排序
    matches = sorted(matches, key = lambda x:x.distance) 
    # 提取前一百的匹配点

    img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:100],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) 
    plt.imshow(img3)
    plt.show()
    return kp1, kp2
# 基于SIFT的特征匹配

def rebuild(xy_l,xy_r,fin_l,fin_r):
    i=0
    num=len(xy_l)
    R_l=calcuR(fin_l[3],fin_l[4],fin_l[5])
    R_r=calcuR(fin_r[3],fin_r[4],fin_r[5])
    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    while i<num:
        A=[]
        b=[]
        l1=f*R_l[0]+(xy_l[i][0]-x0)*R_l[2]
        l2=f*R_l[3]+(xy_l[i][0]-x0)*R_l[5]
        l3=f*R_l[6]+(xy_l[i][0]-x0)*R_l[8]
        lx=f*R_l[0]*fin_l[0]+f*R_l[3]*fin_l[1]+f*R_l[6]*fin_l[2]+(xy_l[i][0]-x0)*R_l[2]*fin_l[0]+(xy_l[i][0]-x0)*R_l[5]*fin_l[1]+(xy_l[i][0]-x0)*R_l[8]*fin_l[2]
        l4=f*R_l[1]+(xy_l[i][1]-y0)*R_l[2]
        l5=f*R_l[4]+(xy_l[i][1]-y0)*R_l[5]
        l6=f*R_l[7]+(xy_l[i][1]-y0)*R_l[8]
        ly=f*R_l[1]*fin_l[0]+f*R_l[4]*fin_l[1]+f*R_l[7]*fin_l[2]+(xy_l[i][1]-y0)*R_l[2]*fin_l[0]+(xy_l[i][1]-y0)*R_l[5]*fin_l[1]+(xy_l[i][1]-y0)*R_l[8]*fin_l[2]
        s1=[l1,l2,l3]
        s2=[l4,l5,l6]
        A.append(s1)
        A.append(s2)
        b.append(lx)
        b.append(ly)
        l1=f*R_r[0]+(xy_r[i][0]-x0)*R_r[2]
        l2=f*R_r[3]+(xy_r[i][0]-x0)*R_r[5]
        l3=f*R_r[6]+(xy_r[i][0]-x0)*R_r[8]
        lx=f*R_r[0]*fin_r[0]+f*R_r[3]*fin_r[1]+f*R_r[6]*fin_r[2]+(xy_r[i][0]-x0)*R_r[2]*fin_r[0]+(xy_r[i][0]-x0)*R_r[5]*fin_r[1]+(xy_r[i][0]-x0)*R_r[8]*fin_r[2]
        l4=f*R_r[1]+(xy_r[i][1]-y0)*R_r[2]
        l5=f*R_r[4]+(xy_r[i][1]-y0)*R_r[5]
        l6=f*R_r[7]+(xy_r[i][1]-y0)*R_r[8]
        ly=f*R_r[1]*fin_r[0]+f*R_r[4]*fin_r[1]+f*R_r[7]*fin_r[2]+(xy_r[i][1]-y0)*R_r[2]*fin_r[0]+(xy_r[i][1]-y0)*R_r[5]*fin_r[1]+(xy_r[i][1]-y0)*R_r[8]*fin_r[2]
        s1=[l1,l2,l3]
        s2=[l4,l5,l6]
        A.append(s1)
        A.append(s2)
        b.append(lx)
        b.append(ly)
        rebuild_xyz=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(A),A)),np.transpose(A)),b)
        ax1.scatter3D(rebuild_xyz[0],rebuild_xyz[1],rebuild_xyz[2], cmap='Blues')  #绘制散点图
        i=i+1
    plt.show()
    return 0

Pointimg=cv.imread('kpnum.jpg') 
iniimg_l=cv.imread('l.bmp') 
iniimg_r=cv.imread('r.bmp') 
# img1=cv.cvtColor(iniimg_l, cv.COLOR_BGR2GRAY)
# img2=cv.cvtColor(iniimg_r, cv.COLOR_BGR2GRAY)
# kp1,kp2=SIFT(img1,img2)
 
P_coor=readTxt("Control_Point_coordinate.txt")
P_coor=np.reshape(P_coor,(60,5))
P_coor=np.delete(P_coor,[0,4],axis=1)
Point_num=GetPointnum(Pointimg) 
P=ControlPointdetect(iniimg_l,Point_num,P_coor) 
xy,XY_3,P_temp=outputxyandXY_3(P,iniimg_l)
h=DLTcalcu(xy,XY_3,iniimg_l)
omega,kapa,phi,XYZs=Externalele(h,x0,y0,f)
fin=E_ele_calcu(omega,kapa,phi,XYZs,xy,XY_3,f,x0,y0)
P0=ControlPointdetect(iniimg_r,Point_num,P_coor) 
xy0,XY_30,P_temp0=outputxyandXY_3(P0,iniimg_r)
h0=DLTcalcu(xy0,XY_30,iniimg_l)
omega0,kapa0,phi0,XYZs0=Externalele(h0,x0,y0,f)
fin0=E_ele_calcu(omega0,kapa0,phi0,XYZs0,xy0,XY_30,f,x0,y0)

xy_l,xy_r,XYZ=thesamepoint(P_temp,P_temp0)
res=PointProjectionele(fin0,fin,xy_r,xy_l,XYZ,f)
S=3
k=1
sigma=1.6
xy_matching_l,xy_matching_r=digital.Pointserch('l.bmp','r.bmp',S,k,sigma)
rebuild(xy_matching_l,xy_matching_r,fin,fin0)


