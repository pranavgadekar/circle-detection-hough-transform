from __future__ import division
import cv2
import numpy as np
import random
import math
import time


original_image = cv2.imread('Sample_Input.jpg',1)
#cv2.imshow('Gray Image',original_image)

#Gaussian Blurring of Gray Image
blur_image = cv2.GaussianBlur(original_image,(3,3),0)
#cv2.imshow('Gaussian Blurred Image',blur_image)

#Using OpenCV Canny Edge detector to detect edges
edged_image_one = cv2.Canny(original_image,75,150)
edged_image_two = cv2.Canny(original_image,75,150)
#cv2.imshow('Edged Image One', edged_image_one)

height,width = edged_image_two.shape
print height,width
#finding contours from a image
contours, hierarchy = cv2.findContours(edged_image_one,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

V = np.zeros((height,width))
V_list = []     #list used to update set of pixels after each 4 pixel iteration
listV = []      #list to remove all points of true circle from V
f=0     #failure counter
Tf=10    #number of failures
Tmin=60     #minumum number of pixels left in V
Ta=3       #minimum distance between 2 pixels in V
Td=1        #distance threshold for 4th pizel
Tr=60/100   #ratio pixel

edge_pixels = np.where(edged_image_two == 255)
for i in xrange(0,len(edge_pixels[0])):
    x=edge_pixels[0][i]
    y=edge_pixels[1][i]
    V[x][y]=1


def main():
    for i in xrange(0,len(contours)):
        f=0
        circle_detected=0
        if(len(contours[i])>100):
            while (f<=Tf):
                f+=1
                random_pixels=random.sample(contours[i],4)
                x1 = random_pixels[0][0][1]
                y1 = random_pixels[0][0][0]
                x2 = random_pixels[1][0][1]
                y2 = random_pixels[1][0][0]
                x3 = random_pixels[2][0][1]
                y3 = random_pixels[2][0][0]
                x4 = random_pixels[3][0][1]
                y4 = random_pixels[3][0][0]
                
                V[x1][y1] = 0
                V[x2][y2] = 0
                V[x3][y3] = 0
                V[x4][y4] = 0
            
                colinearity = np.absolute(((x2-x1)*(y3-y1))-((x3-x1)*(y2-y1)))
                pixel_dist = check_pixel_distance(x1,y1,x2,y2,x3,y3)

                if(pixel_dist==1 and colinearity!=0):
                    circle_detected=determine_possible_circle(x1,y1,x2,y2,x3,y3,x4,y4,colinearity)
                if(circle_detected==1):
                    break  
               
            
            
           

#function to determine possible circle
def determine_possible_circle(x1,y1,x2,y2,x3,y3,x4,y4,colinearity):
    var1 = x2**2+y2**2 - (x1**2+y1**2)
    var2 = x3**2+y3**2 - (x1**2+y1**2)
    X_center = int(((var1*2*(y3-y1)) - (var2*2*(y2-y1)))/(4*colinearity))
    Y_center = int(((var2*2*(x2-x1)) - (var1*2*(x3-x1)))/(4*colinearity))
    #print X_center,Y_center
    if(X_center>0 and Y_center>0 and X_center<height and Y_center<width):
        radius = int(math.sqrt((x1-X_center)**2 + (y1-Y_center)**2))
        if(radius<70 and radius>20):
            #print radius
            d4 = int(math.sqrt((x4-X_center)**2 + (y4-Y_center)**2))-radius
            if(d4<=Td):
                cv2.circle(original_image,(Y_center,X_center),radius,(255,0,0),2)
                return 1
            else:
                return 0
           
            
            
#function to determine if points are closer than min threshold Ta
def check_pixel_distance(x1,y1,x2,y2,x3,y3):
    d1 = math.sqrt(((x2-x1)**2)+((y2-y1)**2))
    d2 = math.sqrt(((x3-x2)**2)+((y3-y2)**2))
    d3 = math.sqrt(((x3-x1)**2)+((y3-y1)**2))

    if(d1>Ta and d2>Ta and d3>Ta):
        return 1
    else:
        return 0            
            



start_time = time.time()
main()
cv2.imshow('Detected Circle',original_image)
end_time = time.time()
time_taken = end_time-start_time
print 'Time taken for execution',time_taken
cv2.waitKey(0)
cv2.destroyAllWindows()