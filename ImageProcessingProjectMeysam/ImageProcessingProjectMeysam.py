#In The Name Of GOD
#Meysam Khodarahmi student no:40034264
#detect cola bottle situations
import numpy as np
from BottleCap import *
import numpy as np
#from BottleCap import *
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
#متد اجرای متدهای شناسایی بر روی تصویر و نمایش نتیجه نهایی
def  find_faults(I):
    fault_list= np.empty((6,2), dtype='S256')#np.zeros((6,2))
    fault_list[0,0] = "Cap: ";
    fault_list[1,0] = "Filled Level: ";
    fault_list[2,0] = "Label: ";
    fault_list[3,0] = "Label Print: ";
    fault_list[4,0] = "Label Straight: ";
    fault_list[5,0] = "Deformed: ";
    
    # cap missing
    bottlecap=BottleCap(I)
    cap_missing = bottlecap.BottleCapCheck();
    if (cap_missing == 0):
        fault_list[0,1] = "Cap Present";
    else:
        fault_list[0,1] = "Cap Missing";
     
    
    # filled level
    filled_level = bottlecap.BottleFilledCheck()
    if (filled_level  == 1):
        fault_list[1,1] = "Over-filled";
    elif( filled_level == 0):
        fault_list[1,1] = "Normal-filled";
    else:
        fault_list[1,1] = "Under-filled";
    
       
    # label missing
    label_missing = bottlecap.BottleLabelCheck();
    if (label_missing == 1):
        fault_list[2,1] = "Label missing";
    else: 
        fault_list[2,1] = "Label Present";
     
    # label print
    if (label_missing == 0):
        
        label_print = bottlecap.BottleLabelPrint() ;
        if (label_print == 1):
            fault_list[3,1] = "Label Print Missing";
        else:
            fault_list[3,1] = "Label Print Present";
         
    else:
        fault_list[3,1] = "NULL since no Label";
     
                    
    # label straight
    if (label_missing == 0):
        
        label_straight = bottlecap.BottleLabelStraight();
        if (label_straight == 1):
            fault_list[4,1] = "Label Not Straight";
        else:
            fault_list[4,1] = "Label Straight";
         
    else:
        fault_list[4,1] = "NULL since no Label";
     
    
    # deformed
    
    if (filled_level == -1):
        fault_list[5,1] = "Uncertain since Underfilled";
    else:
        is_deformed = bottlecap.BottleDeformedCheck();
        if (is_deformed == 1):
            fault_list[5,1] = "Deformed";
        else:
            fault_list[5,1] = "Not Deformed";
    return fault_list     
    
    

#from BottleCap import *

#path = "G:\\bugeto\\پردازش  تصویر\\ImageProcessingProjectMeysam\\ImageProcessingProjectMeysam\\ImageProcessingProjectMeysam\\TrainingData\\Normal\\normal-image1.jpg"
#image=mpimg.imread(path)
#pathLabelMissing = "G:\\bugeto\\پردازش  تصویر\\ImageProcessingProjectMeysam\\ImageProcessingProjectMeysam\\ImageProcessingProjectMeysam\\\TrainingData\\3-NoLabel\\nolabel-image1.jpg"
#image=mpimg.imread(pathLabelMissing)
#pathLabelPrintMissing = "G:\\bugeto\\پردازش  تصویر\\ImageProcessingProjectMeysam\\ImageProcessingProjectMeysam\\ImageProcessingProjectMeysam\\\TrainingData\\4-NoLabelPrint\\nolabelprint-image1.jpg"
#image=mpimg.imread(pathLabelPrintMissing)  
#pathLabelstraight = "G:\\bugeto\\پردازش  تصویر\\ImageProcessingProjectMeysam\\ImageProcessingProjectMeysam\\ImageProcessingProjectMeysam\\\TrainingData\\5-LabelNotStraight\\labelnotstraight-image1.jpg"
#image=mpimg.imread(pathLabelstraight)   
#pathLabeldeformed = "G:\\bugeto\\پردازش  تصویر\\ImageProcessingProjectMeysam\\ImageProcessingProjectMeysam\\ImageProcessingProjectMeysam\\\TrainingData\\7-DeformedBottle\\deformedbottle-image001.jpg"
#image=mpimg.imread(pathLabeldeformed)   

#plt.imshow(image)
#plt.show()

#bottlecap=BottleCap(image)

#cap_missing = bottlecap.BottleCapCheck();
#cap_missing

#filled_level=bottlecap.BottleFilledCheck()
#filled_level


#label_missing = bottlecap.BottleLabelCheck()
#label_missing

#label_print = bottlecap.BottleLabelPrint()
#label_print
#label_straight = bottlecap.BottleLabelStraight();
#label_straight


#is_deformed = bottlecap.BottleDeformedCheck();
#is_deformed
path = "TrainingData\\Normal\\normal-image1.jpg"
image=mpimg.imread(path)
fault_list=find_faults(image)
fault_list

pathLabelMissing = "TrainingData\\3-NoLabel\\nolabel-image1.jpg"
image=mpimg.imread(pathLabelMissing)
fault_list=find_faults(image)
fault_list

pathLabelPrintMissing = "TrainingData\\4-NoLabelPrint\\nolabelprint-image1.jpg"
image=mpimg.imread(pathLabelPrintMissing) 
fault_list=find_faults(image)
fault_list


pathLabelstraight = "TrainingData\\5-LabelNotStraight\\labelnotstraight-image1.jpg"
image=mpimg.imread(pathLabelstraight)  
fault_list=find_faults(image)
fault_list



pathLabeldeformed = "TrainingData\\7-DeformedBottle\\deformedbottle-image001.jpg"
image=mpimg.imread(pathLabeldeformed)  
fault_list=find_faults(image)
fault_list