import os, glob, math
from PIL import Image
import numpy as np
import cv2

Patch_size =16
Scale = 2;
Stride = 16;

def Generate_testing_data(data_path):
    # Read *.bmp file
    Img_data = glob.glob(os.path.join(data_path, "*.bmp"))

    for Img_tmp in Img_data:
        # Read each image and convert the rgb image to the gray image using 'LA'
        Img = Image.open(Img_tmp).convert('L')
        Height, Width = np.size(Img)

        # Image cropping
        [LR_Height, LR_Width] = [math.floor(Height / Scale), math.floor(Width / Scale)]
        [HR_Height, HR_Width] = [Scale * LR_Height, Scale * LR_Width]
        Img = Img.crop((0, 0, HR_Height, HR_Width))

        # Convert PIL.Image type to np.array type
        Img = np.array(Img, dtype=np.double)

        # Downsize HR Image to LR Image using bicubic Interpolation
        LR_Img = cv2.resize(Img, dsize=((LR_Height, LR_Width)), interpolation=cv2.INTER_CUBIC)
        LR_Img = cv2.resize(LR_Img, dsize = ((HR_Height, HR_Width)), interpolation=cv2.INTER_CUBIC)

    return HR_Height, HR_Width,  LR_Img/255 , Img/255