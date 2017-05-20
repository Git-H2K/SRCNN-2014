import os, glob, math
from PIL import Image
import numpy as np
import cv2

Data_path = "./train/"
Patch_size = 33
Scale = 2;
Stride = 14;
rot_ang_num = 4;

def Extract_subimg(Img,Patch_size,stride):
    Height,Width  = np.shape(Img)

    Index1 = range(0,Height-Patch_size,stride)
    Index2 = range(0,Width-Patch_size,stride)
    Output = np.zeros([Patch_size*Patch_size,len(Index1)*len(Index2)])
    count = 0

    for w in Index2:
        for h in Index1:
            temp = Img[h:h+Patch_size,w:w+Patch_size]
            Output[:,count]= np.reshape(temp,[1,Patch_size*Patch_size])
            count = count+1
    return Output

def Generate_training_data(data_path):
    # Read *.bmp file
    Img_data = glob.glob(os.path.join(data_path, "*.bmp"))
    Count = 0;
    Vec_LR_Save = np.empty((Patch_size*Patch_size,0), float)
    Vec_HR_Save = np.empty((Patch_size*Patch_size,0), float)
    for Img_tmp in Img_data:

        # Read each image and convert the rgb image to the gray image using 'LA'
        Img=Image.open(Img_tmp).convert('L')
        Height,Width = np.size(Img)

        # Image cropping
        [LR_Height, LR_Width] = [math.floor(Height/Scale), math.floor(Width/Scale)]
        [HR_Height, HR_Width] = [Scale*LR_Height, Scale*LR_Width]
        Img = Img.crop((0,0,HR_Height,HR_Width))

        # Convert PIL.Image type to np.array type
        Img = np.array(Img, dtype=np.double)

        for rot_ang in range(rot_ang_num):

            # Data Augmentation
            HR_Aug_Img = np.rot90(Img,rot_ang)
            HR_Aug_Img_Size = np.shape(HR_Aug_Img)

            HR_Height = HR_Aug_Img_Size[1]
            HR_Width = HR_Aug_Img_Size[0]

            [LR_Height, LR_Width] = [math.floor(Height/Scale), math.floor(Width/Scale)]

            # Downscale HR Image to LR Image using bicubic Interpolation
            LR_Aug_Img = cv2.resize(HR_Aug_Img, dsize = ((LR_Height, LR_Width)), interpolation=cv2.INTER_CUBIC)
            LR_Aug_Img = cv2.resize(LR_Aug_Img, dsize = ((HR_Height, HR_Width)), interpolation=cv2.INTER_CUBIC)

            # Vectorize HR Image and LR Image
            Vec_LR = Extract_subimg(LR_Aug_Img,Patch_size,Stride)
            Vec_HR = Extract_subimg(HR_Aug_Img,Patch_size,Stride)

            # Concatenate LR patches and HR patches
            Vec_LR_Save = np.concatenate((Vec_LR_Save, Vec_LR),axis=1)
            Vec_HR_Save = np.concatenate((Vec_HR_Save, Vec_HR),axis=1)


    vec_size = np.shape(Vec_HR_Save)

    # Shuffle each LR and HR patches
    Rand_idx = np.random.permutation(vec_size[1])

    # Normalize
    Vec_LR_Save = Vec_LR_Save[:,Rand_idx]/255.0
    Vec_HR_Save = Vec_HR_Save[:,Rand_idx]/255.0

    # Vec_LR_save shape = [:,33,33,1]
    Vec_LR_Save = np.reshape(Vec_LR_Save,[Patch_size,Patch_size,1,vec_size[1]])
    Vec_LR_Save=Vec_LR_Save.swapaxes(1,2)
    Vec_LR_Save=Vec_LR_Save.swapaxes(0,1)
    Vec_LR_Save=Vec_LR_Save.swapaxes(0,3)

    # Vec_LR_save shape = [:,21,21,1]
    Vec_HR_Save = np.reshape(Vec_HR_Save,[Patch_size,Patch_size,1,vec_size[1]])
    Vec_HR_Save=Vec_HR_Save.swapaxes(1,2)
    Vec_HR_Save=Vec_HR_Save.swapaxes(0,1)
    Vec_HR_Save=Vec_HR_Save.swapaxes(0,3)
    Vec_HR_Save = Vec_HR_Save[:,6:27,6:27,:]

    return Vec_LR_Save,Vec_HR_Save
