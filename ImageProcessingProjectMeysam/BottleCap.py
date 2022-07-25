#In The Name Of GOD
#Meysam Khodarahmi student no:40034264
#Identify the appearance of the cola bottle 
from tkinter import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from skimage import morphology
from skimage.measure import *
from PIL import Image as img, ImageFilter
from skimage.measure import label, regionprops, regionprops_table
import pandas as pd
import scipy
import math
#main class
class BottleCap(object):
    def __init__(self,image):
        self._Image=image
    @property
    def get_Image(self):
        return self._Image
     
    def set_Image(self,value):
        self._Image=value
        self._Image=value
    #convert to grayscale
    def rgb_to_gray(self,img):
        grayImage = np.zeros(img.shape)
        R = np.array(img[:, :, 0])
        G = np.array(img[:, :, 1])
        B = np.array(img[:, :, 2])

        R = (R *.299)
        G = (G *.587)
        B = (B *.114)

        Avg = (R+G+B)
        grayImage = img.copy()

        for i in range(3):
           grayImage[:,:,i] = Avg
           
        return grayImage 
    #detect if the bottle has cap or does not have
    def  BottleCapCheck(self):
          I=self._Image
          #converting RGB to grayscale 
          I_gray = self.rgb_to_gray(I);
          
          #getting the original dimensions of input image
          #به دست آوردن ابعاد تصویر ورودی  
          original_ylen = I_gray.shape[0];
          original_xlen = I_gray.shape[1];
          #ساخت پنجره فوکوس روی درب نوشابه
          #creating a focus window
          window_y = math.ceil(0.18 * original_ylen);
          window_x = math.ceil(0.33 * original_xlen);
          
          #populating the cap area of image
          #ساخت یک آرایه به اندازه منطقه درب نوشابه
          cap_region = np.zeros((window_y, window_x,3));
          
          #assigning the coordinates of the area of focus
          #منطقه درب نوشابه را با توجه با ابعاد تصویر مشخص میکنیم
          x1 = math.ceil(original_xlen/3);
          x2 = x1+window_x;
          x3 = x1;
          x4 = x3+window_x;
          
          y1 = 0;
          y2 = 1;
          y3 = window_y;
          y4 = window_y;
          
          m = 0;
          n = 0;
          
          #populating the cap_region from the original image
          #مقدار دهی آرایه به سایز بخش درب نوشابه از روی تصویر اصلی
          for i in range( y1,y3):
              n = 0;
              for j in range (x1,x2):
                  cap_region[m,n] = I_gray[i,j];
                  n = n+1;
              m = m+1;
           
          
          #converting the "double" cap_region to "int"
          #تبدیل مقادیر اعشاری به صحیح
          cap_region = cap_region.astype(int) ;
          
          #finding sum of the values within the cap_region
          #مجموع مقادیر پیکسلها در منطقه درب نوشابه را به دست می آوریم
          s = sum(sum(cap_region));
          res = 0
          #if more white, sum is more, hence cap is missing
          #if more black, sum is less, hence cap is present
          #اگر مجموع روشنایی پیکسلها از حد آستانه مشخص شده بیشتر باشد یعنی درب نوشابه وجود ندارد و اگر کمتر باشد یعنی تیره است و درب وجود دارد  
          if (s.all() > 1170000):
              res = 1;
          else:
              res = 0;
          return res
    #جدا کردن بخش مورد نظر از تصویر
    def  CropImage(self,I, x, y, len_x, len_y):
        #is used to crop a given image
        #x,y is the coordinate to start the crop
        #len_x and len_y is the lengths of the cropped section of image
    
        crop = np.zeros((len_y,len_x,3));
    
        m = 0;
        n = 0;
    
        for i in range( y,y+len_y):
            n = 0;
            for j in range( x,x+len_x):
                crop[m,n] = I[i,j];
                n = n+1;
            m = m+1;
    
        crop =  crop.astype(int);
        return crop
    #محاسبه هیستوگرام تصویر 
    def  Calculate_Histogram(self,I):
        hist = np.zeros((1,256 ));
        [r,c,z] = I.shape;
    
        for i in range( 0,r-1):
            for j in range( 0 ,c-1):
                cur_val = I[i,j];
                hist[0,cur_val] = hist[0,cur_val ] + 1;
        return hist    
    #محاسبه جمع مقادیر هیستوگرام    
    def SumHistogram(self,h, start, ending):
        sum = 0;
        if (start < 0 or ending > 255):
            sum = -1;
        else:
            for i in range( start,ending):
                sum = sum + h[0,i+1];
        return sum    
           
 
    #متد بررسی اینکه آیا نوشابه پر است یا نه
    def  BottleFilledCheck(self):
         I=self._Image
            
         #converting RGB to grayscale
         #تبدیل تصویر به خاکستری
         I_gray = self.rgb_to_gray(I);
    
         #getting the original dimensions of input image
         #به دست آوردن ابعاد تصویر اصلی
         original_ylen = I_gray.shape[0];
         original_xlen = I_gray.shape[1];
         
         #creating a focus window
         #ساختن پنجره فوکوس در قسمت بالای نوشابه   
         window_y = 115;
         window_x = 120;
         
         x_start = math.ceil(original_xlen/3);
         y_start = 60;
         
         #getting a cropped image of specified dimensions
         #جداکردن بخش بالای تصویر مربوط به قسمت بالای قوطی نوشابه   
         cropped = self.CropImage(I_gray,x_start,y_start,window_x,window_y);
         
         #finding the histogram
         #پیدا کردن هیستوگرام قسمت بالای قوطی نوشابه    
         hist = self.Calculate_Histogram(cropped);
         
         #calculating the sum of histogram in the black region (150 pixel value)
         #محاسبه مجموع مقادیر هستوگرام نواحی تیره قسمت بالای قوطی نوشابه با روشنایی کمتر از 150   
         hist_sum_150 = self.SumHistogram(hist,0 ,150);
         
         #if sum is small, implies underfilled
         #if sum is large, implies overfilled
         #if sum is in normal range of 3500 - 5500, implies normal
         #اگر مجموع هستوگرام نواحی تیره کمتر از 3500 باشد یعنی نوشابه کمتر از حد معمول پر شده است   
         #اگر مجموع هستوگرام نواحی تیره بیشتر از 5500 باشد یعنی نوشابه بیشتر از حد معمول پر شده است   
         #اگر مجموع هستوگرام نواحی تیره بیشتر از 3500 و کمتر از 5500 باشد یعنی نوشابه به حد معمول پر شده است   
         if (hist_sum_150 < 3500):
             res = -1;
         elif( hist_sum_150 > 5500):
             res = 1;
         else:
             res = 0;
         return res
    #متد کنترل برچسب قوطی نوشابه
    def  BottleLabelCheck(self):

        #using green channel, since easier to differentiate
        #به دلیل آسانتر بودن کار از کانال سبز تصویر استفاده می شود
        I=self._Image
        I_red = I[:,:,0];
        plt.imshow(I_red)
        plt.show()
        #جدا کردن منطقه مربوط به برچشب قوطی نوشابه
        cropped = self.CropImage(I_red, 115,180, 125, 106);
        plt.imshow(cropped)
        plt.show()
        #محاسبه هیستوگرام منطقه برچسب
        hist = self.Calculate_Histogram(cropped);
        #محاسبه جموع هیستوگرام برای نقاط تیره و نقاط روشن برچسب قوطی نوشابه
        s_black = self.SumHistogram(hist, 0, 150);
        s_all = self.SumHistogram(hist, 0, 255);
        #s_white = sum(imhist(cropped)) - s_black ;
        s_white = s_all - s_black ;
        #اگر مجموع هیستوگرام نواحی تیره بیشتر باشد یعنی برچسب ندارد و در غیر اینصورت برچسب دارد
        if (s_black > s_white):
            res = 1;
        else:
            res = 0;
        return res
    #متد مربوط به چک کردن چاپ روی برچسب بطری نوشابه
    def  BottleLabelPrint(self):
    
        #using red channel, since easier to differentiate
        #استفاده از کانال قرمز تصویر به دلیل آسانی کار
        I=self._Image
        I_red = I[:,:,1];
        plt.imshow(I_red)
        plt.show()
        x=int(I.shape[1]/3)
        #جدا کردن منطقه برچسب بطری
        cropped = self.CropImage(I_red, 115,180, 125, 106);
        plt.imshow(cropped)
        plt.show()
        h = self.Calculate_Histogram(cropped);
        #محاسبه مجموع هیستوگرام نقاط با روشنایی نسبتا کم نسبت به مجموع روشنایی همه نقاط
        h_0_200 = self.SumHistogram(h,0,200);
        h_total = self.SumHistogram(h,0,255);
       
        #we are going to consider the cropped image as 'labelprint missing' if
        #percentage of black color pixels (from 0-200) is less than 30% and 'labelprint present' if
        #more than 30%
        #اگر نسبت روشنایی نقاط تیره به کل نقاط کمتر از 30 درصد باشد یعنی چاپ برچسب وجود دارد و در غیر انصورت چاپ وجود ندارد
        if (h_0_200/h_total < 0.30):
            res = 1;
        else:
            res = 0;
        return res 
    #متد آستانه گذاری روی تصویر
    def apply_threshold(self,I, val):
        
        [r,c,z] =  I.shape;
        
        O = np.zeros((r,c));
        
        for i in range( 1, r):
            for j in range( 0,c):
                if (I[i,j,0]  > val):
                    O[i,j] = 255;
                else:
                    O[i,j] = 0;
                 
        return O     
         
                
        
   #متد کنترل اینکه برچسب تصویر کج است یا مستقیم
    def BottleLabelStraight(self):
        I=self._Image
        #using green channel, since easier to differentiate
        #برای راحتی کار از کانال سبز استفاده می شود
        I_green = I[:,:,2];
        plt.imshow(I_green)
        plt.show()
        #جدا کردن منطقه برچسب تصویر
        cropped = self.CropImage(I_green, 115,180, 125, 106);
        plt.imshow(cropped)
        plt.show()
        #will show black if region is above threshold value
        # cropped_thresh = apply_threshold(cropped,200);
        #cropped_thresh = self.apply_threshold(cropped,200);
        #آستانه گذاری برای اینکه اگر بالاتر از حد آستانه باشد سیاه نشان دهد
        cropped_thresh = self.apply_threshold(cropped,50);
        plt.imshow(cropped_thresh)
        plt.show()
        #cropping the normal label line
        #جدا کردن منطقه خط برپسب نرمال
        label_line_cropped = self.CropImage(cropped_thresh,0,0,125, 25);
        plt.imshow(label_line_cropped)
        plt.show()
        #get count of white pixels
        #شمارش پیکسلهای سفید
        white_count = sum(sum(label_line_cropped))/255;
        
        white_percentage =( white_count[0] /  (label_line_cropped.shape[0]* label_line_cropped.shape[1]))*100 ;
        #اگر نسبت پیکسلهای سفید به کل تصویر بیش از 22 درصد بود یعنی برچسب مستقیم و درست است
        if (white_percentage > 22):
            res = 0;
        else: 
            res = 1;
        return res   
    #متد به دست آوردن جدول خطوط و زوایای خطوط عمود بر خطها با استفاده از متد هاف 
    def  get_hough_result(self):
            I=self._Image
            I_green = I[:,:,2];
            # Cropping region of interest
            #جدا کردن قسمت بالای قوطی
            edge_cropped = self.CropImage(I_green, 100, 60, 140, 115);
            # Thresholding the cropped image to show the dark region
            #آستانه گذاری  تصویر برا نشان دادن نواحی تیره
            cropped_thresh = edge_cropped < 100;
            #پر کردن سوراخها در تصویر
            im_flood_fill = cropped_thresh.copy()
            h, w = cropped_thresh.shape[:2]
            mask = np.zeros((h + 2, w + 2), np.uint8)
            im_flood_fill = im_flood_fill.astype("uint8")
            cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
            im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
            img_out = cropped_thresh | im_flood_fill_inv
            cropped_filled=img_out
            plt.imshow(cropped_filled)
            plt.show()
            #########################
            # remove small objects
            #تبدیل تصویر به خاکستری و حذف اشیای کوچک
            cropped_filled = cv2.cvtColor(cropped_filled,cv2.COLOR_BGR2GRAY)
            plt.imshow(cropped_filled)
            plt.show()
            cropped_no_small = morphology.remove_small_objects(cropped_filled, min_size=64, connectivity=2)
            plt.imshow(cropped_no_small)
            plt.show()
            # Get the edges of the image
            #به دست آوردن لبه ها با استفاده از فیلتر کنی
            edge_cropped = cv2.Canny(cropped_no_small,50,150,apertureSize = 3)
            plt.imshow(edge_cropped)
            plt.show()
            # Finding the Hough Transform to find lines in image
            # finding peaks in the Hough Transform 
            #lines =cv2.HoughLines(edge_cropped,1,np.pi/180,200)
            ############################
            #به دست آوردن خطوط و زوایا با استفاده از متد هاف
            lines  = cv2.HoughLinesP(edge_cropped, 1, np.pi / 180, 22, None, 0, 0)
            thetaArray=np.zeros((1,len(lines)))
            if lines is not None:
                for i   in range(0, len(lines)):
                    rho = lines[i][0][0]
                    theta = lines[i][0][1]
                    thetaArray[0,i]=theta
                    a = math.cos(theta)
                    b = math.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                    cv2.line(edge_cropped, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
            #cv2.imshow('hough',edge_cropped)
            #cv2.waitKey(0)
             
            thetaArray
            ################################ 
            
     

            angles = np.zeros((1,5))+90;
            #به دست آوردن زوایای بین خطوط 
            [a1, a2, a3, a4, a5] =thetaArray[0][0],thetaArray[0][1],thetaArray[0][2],thetaArray[0][3],thetaArray[0][4]

            angles[0,0] = a1;
            angles[0,1] = a2;
            angles[0,2] = a3;
            angles[0,3] = a4;
            angles[0,4] = a5;
            
            # finding the absolute of the angles
            filtered_angles = abs(angles);
    
            # set values close to 90 to 90
            #تبدیل زوایای نزدیک به 90 درجه به 90 درجه
            for i in range( 1,  filtered_angles.shape[1]):
                    if (abs(filtered_angles[0,i] - 90) < 15):
                        filtered_angles[0,i]= 90;
                    
    
            # getting non 90 values:
            #به دست آوردن زوایای غیر از 90 درجه
            filtered_unique_no_90 = filtered_angles[filtered_angles!=90];
    
       
            #اگر زوایای غیر از 90 درجه صفر بود به دلیل کم پر شدن بطری قادر به تشخیص له شدگی بطری نیستیم
            #اگر تعداد زوایای غیر 90 درجه بین 22 تا 28 بود یعنی بطری تغییر شکل داده و له شده است
            #در غیر اینصورت بطری سالم و نرمال است
            if (len(filtered_unique_no_90)==0):
                    res = -1;
            elif (len(filtered_unique_no_90[filtered_unique_no_90[0] < 22 or filtered_unique_no_90[0] > 28])>0):
                    res = 1;
            else:
                    res = 0;
                 
            return res




    #متد تشخیص تغییر شکل و له شدگی بطری نوشابه
    def BottleDeformedCheck(self):
        
        I=self._Image    
        I_green = I#[:,:,2];
        #     imshow(I);
    
        # Cropping region of interest
        #جدا کردن قسمت بالای بطری نوشابه 
        edge_cropped = self.CropImage(I_green, 100, 60, 140, 115);
        plt.imshow(edge_cropped)
        plt.show()
         
        #پر کردن حفره ها    
        scipy.ndimage.binary_fill_holes (
              edge_cropped, 
              structure =None, 
              output =None,  
              origin =0
              )
        plt.imshow(edge_cropped)
        plt.show()
     
        #آستانه گذاری
        cropped_thresh = edge_cropped < 100;
        width =20
        #حذف حفره های کوچک
        remove_holes = morphology.remove_small_holes(
            cropped_thresh, width ** 3
        )
        width = 20
        #حذف اشیای کوچک
        remove_objects = morphology.remove_small_objects(
            remove_holes, width ** 3
        )
         
       
        # remove small objects
        
        cropped_no_small=remove_objects
        
        # get properties of object
        #به دست آوردن خصوصیات تصویر شامل طول بیشترین قطر و کمترین قطر و مساحت اشیای موجود در تصویر
        
        cropped_no_small=cropped_no_small[:,:,0]
        plt.imshow(cropped_no_small)
        plt.show()
        label_img = label(cropped_no_small)
         
        props = regionprops_table(label_img, properties=('centroid',
                                                  'orientation',
                                                  'major_axis_length',
                                                  'minor_axis_length',
                                                  'area'))
        
        m=pd.DataFrame(props)
        

        # اگر نسبت بزرگترین قطر به کوچکترین قطر بین 3 تا 3.9 بود و مساحت تصویر کمتر از 4000 بود یعنی بطری تغییر شکل پیدا کرده و له شده است
        # اگر نسبت بزرگترین قطر به کوچکترین قطر بین 3 تا 3.9 بود و مساحت تصویر بیشتر از 4000 بود یعنی بطری تغییر شکل پیدا نکرده و سالم ونرمال است
        #در غیر اینصورت از تبدیل هاف استفاده میکنیم تا خطوط و زوایای بین آنها را به دست آورده و بر اساس آن تصمیم گیری کنیم
        if ((m.major_axis_length.iloc[0] / m.minor_axis_length.iloc[0]) > 3 and (m.major_axis_length.iloc[0] / m.minor_axis_length.iloc[0])< 3.9 and m.area.iloc[0] < 4000):
                res = 1;
        elif ((m.major_axis_length.iloc[0] / m.minor_axis_length.iloc[0]) > 3.4 and (m.major_axis_length.iloc[0] / m.minor_axis_length.iloc[0])< 3.9 and m.area.iloc[0] > 4000):
            res = 0;
        else:
            res = self.get_hough_result();
        return res
  
         







    