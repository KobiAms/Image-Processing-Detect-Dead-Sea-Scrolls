
import numpy as np
import cv2
import glob
import csv


# ----------------- read images from directory  ------------------

def read_images_from_dir(path):
    files_names = []
    images = []
    for file in glob.glob(path+"*"):
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if image.shape[0] < image.shape[1]:
            image = np.rot90(image)
        images.append(image)
        files_names.append(file[len(path): -4])
    return files_names, images     


# -------------------- A: Brightness Filter ------------------

def brightness(src, b):
    bright = src + b
    bright[bright > 255] = 255
    bright[bright < 0] = 0
    return bright.astype(np.uint8)

# -------------------- B: Contrast Filter --------------------

def contrast(src, a):
    cont = (src * a)
    cont[cont > 255] = 255
    if cont.max() < 255:
        cont[0,0] = 255
    return cont.astype(np.uint8)

# -------------------- C: Threshold Filter -------------------

def threshold(src, t):
    th, dst = cv2.threshold(src, t, 255, cv2.THRESH_BINARY)
    return dst.astype(np.uint8)

# -------------------- D: Gamma Correction filter ------------

def gamma(src, g):
    dst = 255*((src/255)**(1/g))
    return dst.astype(np.uint8)

# -------------------- E: Dilation and Erosion ------------

kernel7 = np.ones((7, 7)).astype(np.uint8)
kernel7[0:2,0:2] = [[0, 0], [0, 1]]
kernel7[0:2,5:7] = [[0, 0], [1, 0]]
kernel7[5:7,0:2] = [[0, 1], [0, 0]]
kernel7[5:7,5:7] = [[1, 0], [0, 0]]
kernel7 = 1 - kernel7

kernel3 = np.zeros((3,3)).astype(np.uint8)
kernel3[1, :] = 1
kernel3[:, 1] = 1

def dilation(image, iter=1, kernel=kernel7):
    dilate = cv2.dilate(image, kernel, iterations=iter)
    return dilate

def erosion(image, iter=1, kernel=kernel7):
    erode = cv2.erode(image, kernel, iterations=iter)
    return erode

def morphological_edges(image, iter=1, outside=True, kernel = kernel3):
    erode = erosion(image, iter, kernel)
    dilate = dilation(image, iter, kernel)

    out = np.array((1,1))
    if outside:
        out = dilate - erode
    else:
        out = erode - dilate
    return out   

# ------------- function to clear the unusful sides from image -------------

def clear_margins(image):
    
    th_first = 25
    th_second = 175

    out2 = image.copy().astype(np.uint8)
    out3 = image.copy().astype(np.uint8)
    out3*=0

    canny_edge = cv2.Canny(erosion(dilation(image, 7), 8), th_first, th_second, apertureSize=3, L2gradient=False)
    canny_edge = dilation(canny_edge, 10)
    lines = cv2.HoughLinesP(canny_edge, 1, np.pi/180, 240,  minLineLength=1300, maxLineGap=100)
    if lines is None or len(lines) < 4:
        return -1
    rel_lines = lines[0:4].copy()
    rel_lines *= 0
    rel_lines -= 1

    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if np.abs(y1-y2) < 75: 
            cv2.line(out3, (x1, y1), (x2, y2), (255, 0, 0), 3)
            if (y1 < 1000):
                x1_, y1_, x2_, y2_ = rel_lines[0][0]
                if(y1 > y1_ or y1_ == -1):
                    rel_lines[0] = line
            elif (y1 > image.shape[0]-1000):
                x1_, y1_, x2_, y2_ = rel_lines[1][0]
                if(y1 < y1_ or y1_ == -1):
                    rel_lines[1] = line  
        if (np.abs(x1-x2) < 75):
            cv2.line(out3, (x1, y1), (x2, y2), (255, 0, 0), 3)
            if (x1 < 500):
                x1_, y1_, x2_, y2_ = rel_lines[2][0]
                if(x1 > x1_ or x1_ == -1):
                    rel_lines[2] = line
            elif (x1 > image.shape[1]-500):
                x1_, y1_, x2_, y2_ = rel_lines[3][0]
                if(x1 < x1_ or x1_ == -1):
                    rel_lines[3] = line 


    

    x1_, y1_, x2_, y2_ = rel_lines[0][0]
    out2[:y1_, :] = 255
    x1_, y1_, x2_, y2_ = rel_lines[1][0]
    out2[y1_:, :] = 255
    x1_, y1_, x2_, y2_ = rel_lines[2][0]
    out2[:, :x1_] = 255
    x1_, y1_, x2_, y2_ = rel_lines[3][0]
    out2[:, x1_:] = 255

    return out2

# ------------------ Region Filling ---------------------

def region_filling(image):

    scale_percent = 30 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    kernel3 = np.zeros((3,3)).astype(np.uint8)
    kernel3[1, :] = 1
    kernel3[:, 1] = 1

    h, w = resized.shape
    region_fill = np.zeros((h, w)).astype(np.uint8)
    region_fill[:,0] = 255
    region_fill[0,:] = 255
    region_fill[-1,:] = 255
    region_fill[:,-1] = 255

    dst_invert = np.invert(resized)

    while True:
        tmp = cv2.dilate(region_fill, kernel3, iterations=1)
        tmp = (tmp & dst_invert).astype(np.uint8)
        if np.sum(np.abs(region_fill-tmp)) == 0:
            break
        region_fill = tmp
    region = cv2.resize(region_fill, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_AREA)
    return region


# --------------- Find Mask ---------------------

def find_mask(image):

    res = cv2.GaussianBlur(image, (5,5) , cv2.BORDER_DEFAULT)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(500 ,500))
    res = clahe.apply(res)
    ret3, res = cv2.threshold(res,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
    res = contrast(res, 1.3)
    res = cv2.Canny(res, 0, 45, apertureSize=3, L2gradient=False)
    res = dilation(res, 5)
    return region_filling(res)

# --------------- Detect scrolls ---------------

temp=[]
font = cv2.FONT_HERSHEY_SIMPLEX
header = ['x point', 'y point', 'x + w', 'y + h']

def detect_scrolls(image, mask, file_name):

    res = np.zeros((image.shape[0], image.shape[1], 3)).astype('uint8')
    res[:,:,0] = image
    res[:,:,1] = image
    res[:,:,2] = image
    
    contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contourslist=[]
    counter = 0
    # move all x,y,w,h to list for sort
    for c in contours:

        x, y, w, h = cv2.boundingRect(c)
        
        contourslist.append((x, y, w, h))
        
    contourslist = sorted(contourslist ,key=lambda k:(k[0]*2+k[1]*2)*0.5)
    # open csv file
    f = open('./out/bounding_boxes/'+file_name+'.csv', 'w',encoding='UTF8',newline='')
    writer = csv.writer(f)
    writer.writerow(header)
    for c in contourslist:
        
        x=c[0]
        y=c[1]
        w=c[2]
        h=c[3]

        #  take only contours that in the corrent patern
        if w>100 and h>100:
            if((h>w and w*5>h) or (w>h and h*5>w) or h==w) and (x!=0 or y!=0):
                
                counter+=1
                # draw a green rectangle to visualize the bounding rect
                cv2.rectangle(res, (x, y), (x+w, y+h), (0, 0, 255), 4)
                _x=int(x+(w/2)-50)
                _y=int(y+(h/2)+50)
                cv2.putText(res,str(counter),(_x,_y),font, 4 ,(0,0,255),8,cv2.LINE_AA)
                                
                writer.writerow((str(x),str(y),str(w),str(h)))
    f.close()
    return res

# ------------- detect scrolls conrour ------------------

def get_scroll_contour(image, mask, file_name):

    res = np.zeros((image.shape[0], image.shape[1], 3)).astype('uint8')
    res[:,:,0] = image
    res[:,:,1] = image
    res[:,:,2] = image
    
    contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contourslist=[]
    counter = 0
    # move all x,y,w,h to list for sort

    f = open('./out/contours/'+file_name+'.csv', 'w',encoding='UTF8',newline='')
    writer = csv.writer(f)
    for i, contour in enumerate(contours):

        x, y, w, h = cv2.boundingRect(contour)

        #  take only contours that in the corrent patern
        if w>100 and h>100:
            if((h>w and w*5>h) or (w>h and h*5>w) or h==w) and (x!=0 or y!=0):
                counter+=1

                contourslist.append(contour)       

                row = ["contour "+str(counter)]
                for c in contour:
                    row.append((c[0][1],c[0][1]))
                writer.writerow(row)
                cv2.drawContours(res, [contour], 0, (255, 0, 0), 5)

    f.close()

    return res
