import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from skimage import measure
from scipy import ndimage

# For debugging - plotting images individually

data_path = "data"
output_path = "output"
number = 1
img_path = os.path.join(data_path, str(number), "COR_T1_C.nii")
mask_path = os.path.join(data_path, str(number), "mask.nii")

img_data = nib.load(img_path)
img = img_data.get_fdata()
mask_data = nib.load(mask_path)
mask = mask_data.get_fdata()

print("Folder:", number, "Image shape:", img.shape, " Mask shape:", mask.shape, "Mask values:", np.unique(mask))

layer = 5
img_slice = img[:,:,layer-1]
mask_slice = mask[:,:,layer-1]

print(np.unique(mask_slice))

zeros = np.zeros_like(mask_slice)
vessels = (mask_slice==2).astype(int)
carcinoma = (mask_slice==1).astype(int)
mask3 = (mask_slice==3).astype(int)

labels = measure.label(mask_slice, background=0)

plt.subplots(1)
plt.imshow(img_slice.T, cmap="gray")
plt.imshow(np.stack((vessels.T, zeros.T, zeros.T, 0.75*vessels.T), axis=-1))
plt.imshow(np.stack((zeros.T, carcinoma.T, zeros.T, 0.5*carcinoma.T), axis=-1))
plt.imshow(np.stack((zeros.T, zeros.T, mask3.T, 0.5*mask3.T), axis=-1))
plt.show   

plt.subplots(1)
plt.imshow(labels.T, cmap="jet")
plt.imshow(np.stack((zeros.T, mask_slice.T, zeros.T, 0.5*mask_slice.T), axis=-1))
plt.show   
