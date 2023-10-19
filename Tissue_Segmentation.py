# By Abtin Riasatian

# This code takes a directory of thumbnails and saves the segmented masks (with the same size as the thumbnail).
# Each thumbnail is first padded so its width and height becomes a multiple of 1024. Then square tiles of size
# patch_read_size are extracted from the thumbnail, padded and segmented. At the end these tile masks are concatenated
# together to form the final thumbnail mask.

# Two methods of U-Net segmentation, with different backbones, and Otsu segmentation are available.
# The network is trained with thumbnails at the magnification of around 1x.

# The comparison of differenct backbones are available at: https://arxiv.org/pdf/2006.06531.pdf


# Input list ###########################################################################################################
gpu_num = 0 # the gpu device to use, -1 if you want to use cpu

thmb_dir = './thumbnails/' # input image directory
mask_dir = './masks/'      # mask save directory
img_format = 'jpg'         # input image format, must be one of the image formats could be handled by cv2.imread

segmentation_method = 'unet' # 'unet', 'otsu'

unet_backbone = 'mobilenet' # Matters if segmentation_method is 'unet'.
                            # Possible values are: 'mobilenet', 'efficientnetb3', 
                            # 'resnet50', 'densenet121', 'resnext101', 'vgg16'
							
weight_address = './weights/unet_{0}_50ep_Final_Dataset_Augment_Npp_Loss_snsp.h5'.format(unet_backbone)
                            
patch_read_size = 400 # size of the patch that the algorithm divides the thumbnail into and feeds 
					  # to the network. Should be even and less than 1024.
########################################################################################################################


# Imports ##############################################
import os
os.environ['NVIDIA_VISIBLE_DEVICES'] = str(gpu_num)
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
import glob, cv2, pathlib
from tqdm import tqdm
import numpy as np
import segmentation_models as sm
import tensorflow as tf
# ######################################################



def unet_tissue_segmentation_model_init(weight_address, backbone='mobilenet'):
    tf.reset_default_graph()
    model = sm.Unet(backbone)
    model.load_weights(weight_address)
    return model
    
def unet_tissue_segmentation(thmb_dir, mask_dir, model, patch_read_size=400, img_format='jpg'):
    if patch_read_size>=1024:
        raise Exception('''Please set the patch_read_size to a number smaller than 1024, for example 400, 
                        to give room for padding. This will improve the performance of the algorithm.''')

    thmb_address_list = [pathlib.Path(x).as_posix() for x in glob.glob(thmb_dir+'**/*.'+img_format, recursive=True)]

    thmb_dir = pathlib.Path(thmb_dir).as_posix()
    mask_dir = pathlib.Path(mask_dir).as_posix()

    in_patch_pad = (1024 - patch_read_size)//2

    for thmb_address in tqdm(thmb_address_list):
	
		# This part adds padding to the thumbnail so it could be divided into patches with the specified size------
        f_dir = thmb_address.replace(thmb_dir, '')
        thmb = cv2.imread(thmb_address)
        thmb = cv2.cvtColor(thmb, cv2.COLOR_BGR2RGB)
        thmb_h = thmb.shape[0]
        thmb_w = thmb.shape[1]
        bottom_padding_size = (patch_read_size-(thmb_h%patch_read_size))%patch_read_size
        right_padding_size = (patch_read_size-(thmb_w%patch_read_size))%patch_read_size
        padded_thmb = cv2.copyMakeBorder(
            thmb,
            top=0,
            bottom=bottom_padding_size,
            left=0,
            right=right_padding_size,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
		#===========================================================================================================
        

		# This part goes through the patches of the thumbnail and segments each patch. 
        mask_patch_list = []
        for j in range(0, padded_thmb.shape[0], patch_read_size):
            row_mask_patch_list = []

            for i in range(0, padded_thmb.shape[1], patch_read_size):

                
				# This part adds padding to each patch to make its size to 1024, which is the network input size.
				# Experiments show that the network yields better results if some padding is added and the parts 
				# at the patch borders are not needed to be segmented.-----------------------------------------------------
                cur_patch = padded_thmb[max(0, j-in_patch_pad):min((j+patch_read_size+in_patch_pad), padded_thmb.shape[0]), 
                                        max(0, i-in_patch_pad):min((i+patch_read_size+in_patch_pad), padded_thmb.shape[1])]
                                
                top_border = (1024-cur_patch.shape[0])//2
                bottom_bordder = 1024 - (top_border + cur_patch.shape[0])
                left_border = (1024-cur_patch.shape[1])//2
                right_border = 1024 - (left_border + cur_patch.shape[1])
                
                cur_patch = cv2.copyMakeBorder(
                    cur_patch,
                    top=top_border,
                    bottom=bottom_bordder,
                    left=left_border,
                    right=right_border,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[255, 255, 255]
                )
				#===========================================================================================================

                
				# Segmenting the patches using the network -----------------------------------------------------------------
                network_inp = np.array([cur_patch/255])
                cur_patch_mask = model.predict(network_inp)
                
                
                res_patch_st_j = top_border+min(j, in_patch_pad)
                res_patch_st_i = left_border+min(i, in_patch_pad)
                cur_patch_mask = cur_patch_mask[:, res_patch_st_j:res_patch_st_j + patch_read_size, 
                                                   res_patch_st_i:res_patch_st_i + patch_read_size, :]

                
                cur_patch_mask = cur_patch_mask.reshape((patch_read_size, patch_read_size, 1))
				#===========================================================================================================

                row_mask_patch_list.append(cur_patch_mask) # adding the patch to a list to concat them later

            mask_patch_list.append(row_mask_patch_list)
		
		# This part concats the patches together, binarizes it and saves the segmented mask. -------------------------------
        padded_thmb_mask = cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in mask_patch_list])
        padded_thmb_mask = (padded_thmb_mask*255).astype('uint8')
        ret2, padded_thmb_mask_binary = cv2.threshold(padded_thmb_mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        padded_thmb_mask_binary = padded_thmb_mask_binary[0:thmb_h, 0:thmb_w]
        os.makedirs(os.path.dirname(mask_dir+f_dir), exist_ok=True)
        cv2.imwrite(mask_dir+f_dir, padded_thmb_mask_binary)
		#===================================================================================================================



def otsu_tissue_segmentation(thmb_dir, mask_dir, img_format='jpg'):
    
    thmb_dir = pathlib.Path(thmb_dir).as_posix()
    mask_dir = pathlib.Path(mask_dir).as_posix()
    
    thmb_address_list = [pathlib.Path(x).as_posix() for x in glob.glob(thmb_dir+'**/*.'+img_format, recursive=True)]

    for thmb_address in tqdm(thmb_address_list):
        f_dir = thmb_address.replace(thmb_dir, '')
        thmb = cv2.imread(thmb_address)
        gray_thmb = cv2.cvtColor(thmb, cv2.COLOR_BGR2GRAY)
        ret,thresh_thmb = cv2.threshold(gray_thmb,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thmb_mask_binary = (255-thresh_thmb)
        cv2.imwrite(mask_dir+f_dir, thmb_mask_binary)



def segment_tissue(thmb_dir, mask_dir, weight_address, segmentation_method='unet', 
                   unet_backbone='mobilenet', patch_read_size=400, img_format='jpg'):
    
    pathlib.Path(mask_dir).mkdir(parents=True, exist_ok=True)
    
    if segmentation_method=='unet':
        unet_model = unet_tissue_segmentation_model_init(weight_address, unet_backbone)
        unet_tissue_segmentation(thmb_dir, mask_dir, unet_model, patch_read_size, img_format)
        
    elif segmentation_method=='otsu':
        otsu_tissue_segmentation(thmb_dir, mask_dir, img_format)
        
    else:
        raise ValueError('''Possible values for the variable segmentation_method are 'unet' or 'otsu'. ''')
    
    
segment_tissue(thmb_dir, mask_dir, weight_address, segmentation_method='unet', 
                   unet_backbone='mobilenet', patch_read_size=400, img_format='jpg')
