import numpy as np
import nibabel as nib
from nilearn import datasets

class ATLAS():
    def __init__(self):
        self.dataset_files = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        self.mask_img = nib.load(self.dataset_files.maps)

    def Print_ROIs(self):
        for (i,lab) in enumerate(self.dataset_files.labels):
            print(i,lab)

    def Mask(self, selectors):
        lens = len(selectors)
        if lens == 1:
            mask_sel = self.mask_img.get_fdata()==selectors
        else:
            mask_sel = np.zeros(self.mask_img.get_fdata().shape)
            for sel in selectors:
                mask_sel = np.logical_or(mask_sel,(self.mask_img.get_fdata()==sel))

        masker = np.zeros(self.mask_img.get_fdata().shape)
        masker[mask_sel]=1

        mask_return = nib.Nifti1Image(masker,self.mask_img.affine)

        return mask_return

    def Mask_3D(self, selector):
        mask = self.mask_img.get_fdata()==selector
        return mask
