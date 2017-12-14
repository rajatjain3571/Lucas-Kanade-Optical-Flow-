import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import cv2 as cv
import math
import matplotlib.cm as cm
from scipy.signal import convolve2d
import random


# function to calculate 1st order derivative of gaussian
def gaussian(sigma,x,y):
    a= 1/(np.sqrt(2*np.pi)*sigma)
    b=math.exp(-(x**2+y**2)/(2*(sigma**2)))
    c = a*b
    return a*b


## getting kernel from  gaussian for [-1,0,1]
def gaussian_kernel():
    G=np.zeros((5,5))
    for i in range(-2,3):
        for j in range(-2,3):
            G[i+1,j+1]=gaussian(1.5,i,j)
    return G

def Lucas_Kanade_Expand(image):

    w, h = image.shape
    newWidth = int(w * 2)
    newHei = int(h * 2)
    newImage = np.zeros((newWidth,
                         newHei))  # interpolate of image i.e. inserting making image by inserting 0 alternate to every original pixels
    newImage[::2, ::2] = image
    G = gaussian_kernel()
    for i in range(2, newImage.shape[0] - 2, 2):
        for j in range(2, newImage.shape[1] - 2, 2):
            newImage[i, j] = np.sum(newImage[i - 2:i + 3, j - 2:j + 3] * G)  # convolving with gaussian mask

    return newImage

def LK_Expand_Iterative(Img,Level):
    if Level==0:#level 0 means current level i.e. no change
        return Img
    i=0
    newImage=cv.imread(Img,0)
    while(i<Level):
        newImage=Lucas_Kanade_Expand(newImage)
        i=i+1
    return newImage

def Lucas_Kanade_Pyramid_Reduce(img,level):
    print(cv.imread(img).shape)
    I1 = cv.imread(img, 0)
    if level==0: #if level is zero than returing original image as no change
        return I1
    x=0
    while(x<level):
        w,h=I1.shape
        newWidth=int(w/2)
        newHei=int(h/2)
        G=gaussian_kernel()
        newImage = np.ones((newWidth,newHei))
        for i in range(2,I1.shape[0]-2,2):#making image of half size by skiping alternate pixels
            for j in range(2,I1.shape[1]-2,2):
                newImage[int(i/2),int(j/2)]=np.sum(I1[i - 2:i + 3, j - 2:j + 3]*G)#convolving with gaussian mask
        print(newImage.shape)
        x=x+1
        if x==level:
            return newImage

def Lucas_Kanade_Reduce(I1):
    w, h = I1.shape
    newWidth = int(w / 2)
    newHei = int(h / 2)
    G = gaussian_kernel()
    newImage = np.ones((newWidth, newHei))
    for i in range(2, I1.shape[0] - 2, 2):  # making image of half size by skiping alternate pixels
        for j in range(2, I1.shape[1] - 2, 2):
            newImage[int(i / 2), int(j / 2)] = np.sum(I1[i - 2:i + 3, j - 2:j + 3] * G)  # convolving with gaussian mask

    return newImage

def LK_Reduce_Iterative(Img,Level):
    if Level==0:#level 0 means current level i.e. no change
        return Img
    i=0
    newImage=cv.imread(Img,0)
    while(i<Level):
        newImage=Lucas_Kanade_Reduce(newImage)
        i=i+1

    return newImage


def Lucas_Kanade(image1,I1, image2,I2,Level,Reduce_Expand):
    oldframe = cv.imread(image1)
    I11 = cv.cvtColor(oldframe, cv.COLOR_BGR2GRAY)
    #I1=I1.astype(np.float64)
    newframe = cv.imread(image2)
    #I2 = cv.cvtColor(newframe, cv.COLOR_BGR2GRAY)

    color = np.random.randint(0, 255, (100, 3))
    Gx = np.reshape(np.asarray([[-1, 1], [-1, 1]]), (2, 2))  # for image 1 and image 2 in x direction
    Gy = np.reshape(np.asarray([[-1, -1], [1, 1]]), (2, 2))  # for image 1 and image 2 in y direction
    Gt1 = np.reshape(np.asarray([[-1, -1], [-1, -1]]), (2, 2))  # for 1st image
    Gt2 = np.reshape(np.asarray([[1, 1], [1, 1]]), (2, 2))  # for 2nd image


    Ix = (convolve2d(I1, Gx) + convolve2d(I2, Gx)) / 2 #smoothing in x direction

    Iy = (convolve2d(I1, Gy) + convolve2d(I2, Gy)) / 2 #smoothing in y direction
    It1 = convolve2d(I1, Gt1) + convolve2d(I2, Gt2)   #taking difference of two images using gaussian mask of all -1 and all 1

    # parameter to get features
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    features= cv.goodFeaturesToTrack(I11, mask = None, **feature_params)  #using opencv function to get feature for which we are plotting flow
    feature = np.int32(features)

    ## IF we are reducing than fetures will be reduced by 2**Level
    if Reduce_Expand=="Reduce":
        feature=np.int32(feature/(2**Level))
    else:
        feature=np.int32(feature*(2**Level)) #if we are expanding than feature will expand by 2*Level
    # print(feature)
    feature = np.reshape(feature, newshape=[-1, 2])

    u = np.ones(Ix.shape)
    v = np.ones(Ix.shape)
    status=np.zeros(feature.shape[0]) # this will tell change in x,y
    A = np.zeros((2, 2))
    B = np.zeros((2, 1))
    mask = np.zeros_like(oldframe)

    newFeature=np.zeros_like(feature)
    """Assumption is  that all the neighbouring pixels will have similar motion. 
    Lucas-Kanade method takes a 3x3 patch around the point. So all the 9 points have the same motion.
    We can find (fx,fy,ft) for these 9 points. So now our problem becomes solving 9 equations with two unknown variables which is over-determined. 
    A better solution is obtained with least square fit method.
    Below is the final solution which is two equation-two unknown problem and solve to get the solution.
                               U=Ainverse*B 
    where U is matrix of 1 by 2 and contains change in x and y direction(x==U[0] and y==U[1])
    we first calculate A matrix which is 2 by 2 matrix of [[fx**2, fx*fy],[ fx*fy fy**2] and now take inverse of it
    and B is -[[fx*ft1],[fy,ft2]]"""

    for a,i in enumerate(feature):

        x, y = i

        A[0, 0] = np.sum((Ix[y - 1:y + 2, x - 1:x + 2]) ** 2)

        A[1, 1] = np.sum((Iy[y - 1:y + 2, x - 1:x + 2]) ** 2)
        A[0, 1] = np.sum(Ix[y - 1:y + 2, x - 1:x + 2] * Iy[y - 1:y + 2, x - 1:x + 2])
        A[1, 0] = np.sum(Ix[y - 1:y + 2, x - 1:x + 2] * Iy[y - 1:y + 2, x - 1:x + 2])
        Ainv = np.linalg.pinv(A)

        B[0, 0] = -np.sum(Ix[y - 1:y + 2, x - 1:x + 2] * It1[y - 1:y + 2, x - 1:x + 2])
        B[1, 0] = -np.sum(Iy[y - 1:y + 2, x - 1:x + 2] * It1[y - 1:y + 2, x - 1:x + 2])
        prod = np.matmul(Ainv, B)

        u[y, x] = prod[0]
        v[y, x] = prod[1]

        newFeature[a]=[np.int32(x+u[y,x]),np.int32(y+v[y,x])]
        if np.int32(x+u[y,x])==x and np.int32(y+v[y,x])==y:    # this means that there is no change(x+dx==x,y+dy==y) so marking it as 0 else
            status[a]=0
        else:
            status[a]=1 # this tells us that x+dx , y+dy is not equal to x and y

    um=np.flipud(u)
    vm=np.flipud(v)
    if Reduce_Expand=="Reduce":# multiplying by 2**Level to get position in original image
        good_new=np.int32(newFeature[status==1]*(2**Level)) #status will tell the position where x and y are changed so for plotting getting only that points
        good_old = np.int32(feature[status==1]*(2**Level))
        print(good_old)
        print("Good new")

        print(good_new)
    else:#divding by 2**level to get original position back
        good_new = np.int32(newFeature[status == 1] / (2 ** Level))  # status will tell the position where x and y are changed so for plotting getting only that points
        good_old = np.int32(feature[status == 1] / (2 ** Level))


    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        newframe = cv.circle(newframe, (a, b), 5, color[i].tolist(), -1)
    img = cv.add(newframe, mask)
    return img

"""
 Tracking over image pyramids
allows track points of interest despite large disparity between
photos
"""



###### Reduced Image Pyramid###############################

I1_reduce=LK_Reduce_Iterative("basketball1.png",1)
I2_reduce=LK_Reduce_Iterative("basketball2.png",1)
FinalReduceImage1=Lucas_Kanade("basketball1.png",I1_reduce,"basketball2.png",I2_reduce,1,"Reduce")

plt.imshow(FinalReduceImage1)
plt.show()


I3_Reduce=LK_Reduce_Iterative("grove1.png",2)

I4_Reduce=LK_Reduce_Iterative("grove2.png",2)
FinalReduceImage2=Lucas_Kanade("grove1.png",I3_Reduce,"grove2.png",I4_Reduce,2,"Reduce")
plt.imshow(FinalReduceImage2)
plt.show()
#
# ###### Expand Image Pyramid###############################

I1_expand=LK_Expand_Iterative("basketball1.png",1)
I2_expand=LK_Expand_Iterative("basketball2.png",1)
FinalExpandImage1=Lucas_Kanade("basketball1.png",I1_expand,"basketball2.png",I2_expand,1,"Expand")
plt.imshow(FinalExpandImage1)
plt.show()


I3_expand=LK_Expand_Iterative("grove1.png",2)

I4_expand=LK_Expand_Iterative("grove2.png",2)
FinalExpandImage2=Lucas_Kanade("grove1.png",I3_expand,"grove2.png",I4_expand,2,"Expand")
plt.imshow(FinalExpandImage2)
plt.show()










