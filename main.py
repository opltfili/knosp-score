import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from skimage import measure
from scipy import ndimage

data_path = "data"
output_path = "output"

# For each folder/patient
for number in range(1, 15): # number = int(input("Select patient's number (1-14):"))
    img_path = os.path.join(data_path, str(number), "COR_T1_C.nii")
    mask_path = os.path.join(data_path, str(number), "mask.nii")

    # Load data with NiBabel
    img_data = nib.load(img_path)
    img = img_data.get_fdata()
    mask_data = nib.load(mask_path)
    mask = mask_data.get_fdata()

    # Check shapes
    print("Folder:", number, "Image shape:", img.shape, " Mask shape:", mask.shape, "Mask values:", np.unique(mask))

    # For each layer
    for layer in range(img.shape[2]): #layer = int(input("Select layer: (0-"+str(img.shape[2]-1)+")"))
        img_slice = img[:,:,layer]
        mask_slice = mask[:,:,layer]

        zeros = np.zeros_like(mask_slice) # "empty" layer - for empty RGB channel of appropriate shape (w×h)
        vessels = (mask_slice==2).astype(int)
        carcinoma = (mask_slice==1).astype(int)

        labels = measure.label(mask_slice, background=0) # skimage to label each individual area with different label
        label_vals = np.unique(labels) # list all values, i.e. which objects are in the mask
        
        # Approximate with circles - initialize
        centre_of_mass = np.zeros((len(label_vals), 2))
        radius = np.zeros(len(label_vals))

        # Compute
        for val in label_vals:
            if val == 0: # skip label 0 = background
                continue
            centre_of_mass[val, :] = ndimage.center_of_mass(labels==val) # average x and y coordinates of the area
            radius[val] = np.sqrt(np.sum(labels==val)/np.pi) # area = sum of pixels = S = pi × r^2

        if len(label_vals) > 2: # if there are more labels (object not touching each other)
            com_vessels = centre_of_mass[radius!=radius.max(),:] # skip the biggest object - carcinoma
            com_vessels = com_vessels[1:,:] # skip 0 - background
            middle = np.mean(com_vessels, axis=0)
            left = com_vessels[com_vessels[:,0]<=middle[0]] # left vessels - x <= average x
            right = com_vessels[com_vessels[:,0]>middle[0]] # right vessels - x > average x

        # Visualuzations
        fig, ax = plt.subplots(1,2)

        # 1st figure
        ax[0].imshow(img_slice.T, cmap="gray")  # image slice
        ax[0].imshow(np.stack((vessels.T, zeros.T, zeros.T, 0.75*vessels.T), axis=-1)) # vessels in red channel, transparency 25%
        ax[0].imshow(np.stack((zeros.T, carcinoma.T, zeros.T, 0.5*carcinoma.T), axis=-1)) # carcinoma in green channel, transparency 50%
        
        # 2nd figure
        labels_img = ax[1].imshow(labels.T, cmap="jet") # show segmented objects, each with different color
        if len(label_vals) > 2: # if there are more types of segmented objects, not only background and foreground
            for i in range(1,len(label_vals)): # plot circles around vessels - skip background (0)
                if radius[i] == radius.max(): # skip the biggest object - carcinoma
                    continue
                circle = plt.Circle(centre_of_mass[i, :], radius=radius[i], edgecolor="white", facecolor="none", linewidth=0.5)
                ax[1].add_patch(circle) # plot the circle around the vessel
            if len(left) > 1: # if vessels are not connected
                ax[1].axline(left[0], left[1], color="white", linewidth=0.5)
            if len(right) > 1: # if vessels are not connected
                ax[1].axline(right[0], right[1], color="white", linewidth=0.5)
            # plt.colorbar(labels_img, ax=ax)

        # save to output folder
        fig.savefig(os.path.join(output_path, str(number), "layer_"+str(layer+1)+".png"), dpi=300)
        plt.close()
    
    # Print info
    print("Patient", number, "done.")

    # Finish
    print("Done.")
