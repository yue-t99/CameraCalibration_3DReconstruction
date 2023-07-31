from numpy import *
import numpy as np
import sys
import re
import cv2 as cv
from numpy.linalg import  *



def setDspace(img,S,sigma,k):
    Dspace=[]
    Dspacetemp=[]
    Gaussspace=[]
    sigma_list=[]
    imagetemp=img.copy()
    i=n=0
    sigmatemp=0
    while i<k:
        n=0
        Dspacetemp=[]
        while n<(S+3):
            sigmatemp=sigma*(2**(n/S))
            t=cv.GaussianBlur(imagetemp,(0,0),sigmatemp)
            # 依据所给的sigma 进行影像高斯滤波
            # cv.imshow("?",t)
            # cv.waitKey(0)
            Dspacetemp.append(t)
            if(n>1 and n<S+2): 
                temp=Dspacetemp[n]-Dspacetemp[n-1]
                Dspace.append(temp)
                # 生成高斯差分金字塔
            n=n+1
            sigma_list.append(sigmatemp)
        # cv.imshow("tmp",imagetemp)
        # cv.waitKey(0)
        Gaussspace.append(imagetemp)
        imagetemp=cv.pyrDown(imagetemp)
        imagetemp=cv.GaussianBlur(imagetemp,(5,5),0)
        i=i+1    
    i=0
    # while i<S*k:
    #     cv.imshow("test", Dspace[i])
    #     cv.waitKey(0)
    #     i=i+1
    return Dspace,Gaussspace,sigma_list
# 构建尺度空间和高斯金字塔影像

def Pointdetect(Dspace,S,k):
    i=n=0
    Point_re=[]
    while i<k:
        img_range=np.shape(Dspace[(i+1)*S-1])
        point_de=np.zeros(img_range)
        m=n=1
        while m<img_range[0]:
            n=1
            window=[]
            num=0
            while num<S:
                window.append(Dspace[i*S+num][m-1:m+2,0:3])
                num=num+1
            window=np.hstack(window)
            # 设置检测的窗口 大小为3*3*3
            # window=np.hstack(((Dspace[0])[m-1:m+2,0:3],(Dspace[1])[m-1:m+2,0:3],(Dspace[2])[m-1:m+2,0:3]))
            while n<img_range[1]:
                if ((Dspace[i*S+int(S/2)])[m,n]==window.max() and abs((Dspace[i*S+int(S/2)])[m,n])>(0.04*0.5/S)):  #阈值化和极值检测并行处理
                    point_de[m,n]=255
                # 判断是否为极值 并记录判断的极值
                n=n+1
                # Rig_Pointdetect(D,S):
                # window=np.hstack(((Dspace[0])[m-1:m+2,n-1:n+2],(Dspace[1])[m-1:m+2,n-1:n+2],(Dspace[2])[m-1:m+2,n-1:n+2]))
                num=0
                window=[]
                while num<S:
                    window.append(Dspace[i*S+num][m-1:m+2,n-1:n+2])
                    num=num+1
                window=np.hstack(window)
                # 更新窗口
            m=m+1
        i=i+1     
        Point_re.append(point_de)
        # cv.imshow("test",point_de)
        # cv.waitKey(0)
    return Point_re
# 根据极值检测关键点


def Hessian(Point_re,k,img):
    i=0
    KeyPoint=[]
    while i<k:
        imgtemp=Point_re[i]
        H=np.zeros(imgtemp.shape)
        rows,cols = imgtemp.shape[:2]
        for row in range(rows-2):
            for col in range(cols-2):
                if (imgtemp[row+1,col+1]==255):
                    DXX = imgtemp[row+2,col+1]+imgtemp[row+1,col]-2*imgtemp[row+1,col+1]
                    DYY = imgtemp[row,col+1]+imgtemp[row+2,col+1]-2*imgtemp[row+1,col+1]
                    DXY = (imgtemp[row+2,col+2]+imgtemp[row,col]-imgtemp[row+2,col]-imgtemp[row,col+2])/4
                    det_H=DXX*DYY-DXY*DXY
                    Tr_H=DXX+DYY
                    # hessian 矩阵特征向量计算
                    t=(Tr_H*Tr_H)/double(det_H+0.0000001)
                    if(t>1.2 or  det_H<0):
                        H[row+1,col+1]=0
                    else:
                        H[row+1,col+1]=255
                        # img_test=cv.circle(img, (col+1,row+1), 1, (0,0,255), 4)
        KeyPoint.append(H)
        # cv.imshow("test",H)
        # cv.imshow("compare",img_test)
        # cv.waitKey(0)
        i=i+1
    return KeyPoint
#去边缘化效应，得到真正的关键点


def KeyPointdirection(KeyPoint,k,Gaussspace,img,sigma_list,S):
    # sigma=1.0 widowsize=5
    i_num=0
    n=0
    direction=[]
    m_re=[]
    sita_re=[]
    # Gaus=1/273*np.array([[1,4,7],[4,16,26],[7,26,41]])
    while(i_num<k):
        imgtemp=Gaussspace[i_num]
        imgtemp=double(imgtemp)
        kptemp=KeyPoint[i_num]
        m=np.zeros(imgtemp.shape)
        sita=np.zeros(imgtemp.shape)
        dirtemp=np.zeros(imgtemp.shape)
        rows,cols = imgtemp.shape[:2]
        for row in range(rows-2):
            for col in range(cols-2):
                m[row+1,col+1]=sqrt(((imgtemp[row+2,col+1])-imgtemp[row,col+1])**2+(imgtemp[row+1,col+2]-imgtemp[row+1,col])**2)
                sita[row+1,col+1]=math.atan2((imgtemp[row+2,col+1]-imgtemp[row,col+1]),(imgtemp[row+1,col+2]-imgtemp[row+1,col]))
                if(sita[row+1,col+1]>0):
                    sita[row+1,col+1]=sita[row+1,col+1]/math.pi*180
                else:
                    sita[row+1,col+1]=360+sita[row+1,col+1]/math.pi*180
                    if(sita[row+1,col+1]==360):
                        sita[row+1,col+1]=0
        m_re.append(m)
        sita_re.append(sita)
        # 计算每一点的梯度及其角度
        window_size=5
        Gaustemp=cv.getGaussianKernel(window_size,1.5*sigma_list[i_num*(S+3)+n],cv.CV_64F)
        Gaus=Gaustemp*(np.transpose(Gaustemp))
        # 构造窗口 设置大小为5
        for row in range(rows-window_size+1):
            for col in range(cols-window_size+1):
                if(kptemp[row+window_size//2,col+window_size//2]>0):
                    sitatemp=np.zeros(36)
                    for i in range(0,window_size):
                        for j in range(0,window_size):
                            sitatemp[int(sita[row+i,col+j]//10)]=sitatemp[int(sita[row+i,col+j]//10)]+m[row+i,col+j]*Gaus[i,j]
                    dirtemp[row+window_size//2,col+window_size//2]=np.argmax(sitatemp)
                        # cv2.arrowedLine参数概述 
                        # cv2.arrowedLine( 输入图像，起始点(x,y)，结束点(x,y)，线段颜色，线段厚度，线段样式，位移因数， 箭头因数)
        #             img_0=cv.circle(img,(col+window_size//2,row+window_size//2),1,(255,0,0))
        #             img_0 = cv.arrowedLine(img_0, (col+window_size//2,row+window_size//2), \
        #                 (col+window_size//2+int(sitatemp[int(dirtemp[row+window_size//2,col+window_size//2])]*5*cos(dirtemp[row+window_size//2,col+window_size//2]*10)),\
        #                     row+window_size//2+int(sitatemp[int(dirtemp[row+window_size//2,col+window_size//2])]*5*sin(dirtemp[row+window_size//2,col+window_size//2]*10))), \
        #                         (0,0,255),1,8,0,0.3)
        # cv.imshow("test",img_0)
        # cv.waitKey(0)
        # 关键点主方向显示
        direction.append(dirtemp)
        i_num=i_num+1
    return direction,m_re,sita_re
# 为每一个关键点构造一个主方向


def SIFTdiscription(direction,keypoint,k,Gaussspace,sigma,m,sita):
    i_num=0
    n=0
    SIFT_result=[]
    kpindex=[]
    while (i_num<k):
        n=0
        imgtemp=Gaussspace[i_num]
        imgtemp=double(imgtemp)
        rows,cols = imgtemp.shape[:2]
        kptemp=keypoint[i_num]
        window_size=21
        # 构造种子点的窗口 大小为2 9 13 
        r=window_size*4
        SIFT_img=[]
        index=1
        kpindex_temp=[] #np.zeros(imgtemp.shape)
        for row in range(rows-r+1):
            for col in range(cols-r+1):
                if(kptemp[row+r//2,col+r//2]>0 ):
                    kpseed=np.zeros([16,8])
                    main_direction=(direction[i_num])[row+r//2,col+r//2]*10
                    for i in range(0,4):
                        for j in range(0,4):
                            for seed_x in range(0,window_size):
                                for seed_y in range(0,window_size):
                                    lenth=m[i_num][row+i*window_size+seed_x,col+j*window_size+seed_y]
                                    if(sita[i_num][row+i*window_size+seed_x,col+j*window_size+seed_y]-main_direction>0):
                                        angle=sita[i_num][row+i*window_size+seed_x,col+j*window_size+seed_y]-main_direction
                                    else:
                                        angle=360+sita[i_num][row+i*window_size+seed_x,col+j*window_size+seed_y]-main_direction
                                        if(angle==360):
                                            angle=0
                                    kpseed[i*4+j,int(angle//45)]=kpseed[i*4+j,int(angle//45)]+lenth
                    kpseed=np.reshape(kpseed,128)
                    SIFT_img.append(kpseed)
                    kpindex_temp.append([row+r//2,col+r//2])
                    index=index+1
        kpindex.append(kpindex_temp)
        SIFT_result.append(SIFT_img)
        n=n+1
        i_num=i_num+1
    return SIFT_result,kpindex
# SIFT特征向量计算

def SIFT_result(img,S,k,sigma):
    inputimg=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    Dsapce,Gaussspace,sigma_list=setDspace(inputimg,S,sigma,k)
    t=Pointdetect(Dsapce,S,k)
    keypoint=Hessian(t,k,img)
    Direction,m,sita=KeyPointdirection(keypoint,k,Gaussspace,img,sigma_list,S)
    SIFT,kpindex=SIFTdiscription(Direction,keypoint,k,Gaussspace,sigma_list,m,sita)
    return SIFT,kpindex  
# 得到SIFT特征向量

def Hamming_distance(hash1, hash2):
    num = 0
    for index in range(len(hash1)):
        if hash1[index]-hash2[index]>50:
            num = num +1
    return num
# 汉明距离计算

def distance(a,b):
    return np.sqrt(sum(np.power((a - b), 2)))
# 欧式距离计算

def imgstiching(img1,img2):
    r1,c1 = img1.shape[:2]
    r2,c2 = img2.shape[:2]
    reimg1 = np.zeros([max(r1,r2), max(c1,c2), 3], np.uint8)
    reimg2 = np.zeros([max(r1,r2), max(c1,c2), 3], np.uint8)
    for r in range(0,r1) :
        for c in range(0,c1):
            reimg1[r,c]=img1[r,c]
    for r in range(0,r2) :
        for c in range(0,c2):
            reimg2[r,c]=img2[r,c]
    midline=np.zeros([max(r1,r2), 50, 3], np.uint8)
    hmerge=np.hstack((reimg1,midline,reimg2))
    # cv.imshow("test",hmerge)
    # cv.waitKey(0)    
    return hmerge,[max(r1,r2), max(c1,c2)]
# 生成合成影像   

def Pointserch(img_l_path,img_r_path,S,k,sigma):
    img_l=cv.imread(img_l_path)
    img_r=cv.imread(img_r_path)
    img_l=cv.pyrDown(img_l)
    img_l=cv.pyrDown(img_l)
    img_l=cv.pyrDown(img_l)
    img_r=cv.pyrDown(img_r)
    img_r=cv.pyrDown(img_r)
    img_r=cv.pyrDown(img_r)
    hmerge , [r,c] = imgstiching(img_l, img_r) #水平拼接
    l_sift,l_index=SIFT_result(img_l,S,k,sigma)
    r_sift,r_index=SIFT_result(img_r,S,k,sigma)
    temp_1=l_sift
    temp_2=r_sift
    i=j=0
    xy_l=[]
    xy_r=[]
    while i<len(temp_1[0]):
        p_a=temp_1[0][i]
        min_data = {'distance': 10**10, 'point_a': None, 'point_b': None}
        j=0
        for p_b in temp_2[0]:
            dist = distance(p_a, p_b)
            if dist < min_data[ 'distance']:
                min_data['distance'] = dist                
                min_data['point_a'] = l_index[0][i]
                min_data['point_b'] = r_index[0][j]
            j=j+1
        # temp_2[0].pop(r_index[0].index(min_data['point_b']))  
        i=i+int(len(temp_1[0])/200)
        xy_l.append([min_data['point_a'][1]*8,2592-min_data['point_a'][0]*8])
        xy_r.append([min_data['point_b'][1]*8,2592-min_data['point_b'][0]*8])
        # point_s=[int(min_data['point_a'][0]),int(min_data['point_a'][1])]
        # point_e=[int(min_data['point_b'][0]),int(min_data['point_b'][1]+50+c)]
        # if(min_data['distance']<2500): #and abs(min_data['point_a'][0]-min_data['point_b'][0])<50):
        test=cv.line(hmerge,(int(min_data['point_a'][1]),int(min_data['point_a'][0])),\
                (int(min_data['point_b'][1]+50+c),int(min_data['point_b'][0])),(255,0,0),1)
    cv.imshow("test",test)
    cv.waitKey(0)
    return xy_l,xy_r
# 特征点匹配


# S=3
# sigma=1.6
# k=3
# img_l=cv.imread('test1_l.jpg')
# img_r=cv.imread("test1_r.jpg")
# S=3
# k=2
# sigma=1.6
# test=Pointserch('test1_l.jpg',"test1_r.jpg",S,k,sigma)
# 测试代码
