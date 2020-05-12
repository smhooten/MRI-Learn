import numpy as np
import nibabel as nib
from nilearn import image
from nilearn import datasets
from nilearn import masking
from nibabel import processing
from sklearn.model_selection import train_test_split

from ATLAS import ATLAS

import matplotlib.pyplot as plt

class DATA():
    def __init__(self):
        self.Fetch_OASIS()
        self.ROIs_3D_gm = None
        self.ROIs_3D_wm = None

    def Fetch_OASIS(self,balanced=1):
        dataset_files = datasets.fetch_oasis_vbm()
        ## Load datasets
        age = dataset_files.ext_vars['age'].astype(float)
        age = np.array(age)

        gender = dataset_files.ext_vars['mf']
        gender = np.array(gender)
        gender[gender==b'F']=0
        gender[gender==b'M']=1
        gender = gender.astype(float)

        CDR = dataset_files.ext_vars['cdr'].astype(float)
        CDR = np.array(CDR)
        CDR = np.nan_to_num(CDR)

        NWBV = dataset_files.ext_vars['nwbv'].astype(float)
        NWBV = np.array(NWBV)

        ETIV = dataset_files.ext_vars['etiv'].astype(float)
        ETIV = np.array(ETIV)

        ASF = dataset_files.ext_vars['asf'].astype(float)
        ASF = np.array(ASF)

        gm_imgs = np.array(dataset_files.gray_matter_maps)
        wm_imgs = np.array(dataset_files.white_matter_maps)

        features = np.vstack((gender,age,NWBV,ETIV,ASF))
        features = features.T

        idx_CN = np.linspace(0,402,403)[CDR==0].astype('int')
        idx_AD = np.nonzero(CDR)[0]

        labels = -1*np.ones(features.shape[0])
        labels[idx_AD]=1

        self.gm_imgs = gm_imgs
        self.wm_imgs = wm_imgs
        self.age = age
        self.features = features
        self.labels = labels
        self.idx_CN = idx_CN
        self.idx_AD = idx_AD

        if balanced==1:
            self.idx_CN=self.idx_CN[self.age[self.idx_CN]>=59]
            idxs = np.hstack((self.idx_CN,self.idx_AD))
            self.gm_imgs = gm_imgs[idxs]
            self.wm_imgs = wm_imgs[idxs]
            self.features=self.features[idxs,:]
            self.labels = self.labels[idxs]

    def DownSample(epi_img,voxel_sizes):
        #Note epi_img needs to be a Nifti1Image, can use nib.load() to extract
        #Current voxel size 2cm, down sample by setting larger - upsample by setting smaller
        epi_down = processing.resample_to_output(epi_img,voxel_sizes=voxel_sizes)
        return epi_down

    def ScaleDown(epi_img,scale_factor):
        #Note epi_img needs to be a Nifti1Image, can use nib.load() to extract
        #scale_factor: scales image down by factor
        new_affine = np.diag([scale_factor,scale_factor,scale_factor,1])@epi_img.affine #Decrease size by 2x
        org_shape = epi_img.get_fdata().shape #Keep array size the same

        epi_scale = nilearn.image.resample_img(epi_img,target_affine=new_affine, target_shape=org_shape)
        return epi_scale

    def Add_MRI(self,selectors):
        if selectors == "brain":
            masker = masking.compute_gray_matter_mask(self.gm_imgs[0]) #computes full brain mask... misnamed function
        else:
            Atlas = ATLAS()
            masker = Atlas.Mask(selectors)
        gm = masking.apply_mask(self.gm_imgs, masker)
        wm = masking.apply_mask(self.wm_imgs, masker)
        brain = gm+wm
        self.masker = masker
        self.features = np.hstack((self.features,brain))

    def Train_Test(self,train,random=1234):
        #Inputs train size ratio
        #Use k-cross or further split
        idxs = np.arange(0,self.features.shape[0])
        self.idx_train,self.idx_test = train_test_split(idxs, train_size=train, random_state=random)

    def Split_Data(self):
        self.features_train = self.features[self.idx_train]
        self.labels_train = self.labels[self.idx_train]

        self.features_test = self.features[self.idx_test]
        self.labels_test = self.labels[self.idx_test]


    def load_images(self):
        gm_img = []
        n = self.gm_imgs.shape[0]
        for i in range(n):
            img = image.load_img(self.gm_imgs[i])
            gm_img.append(np.array(img._data_cache))
        gm_imgs = np.zeros([n]+list(gm_img[0].shape))
        for i in range(n):
            gm_imgs[i, ...] = gm_img[i]
        
        wm_img = []
        n = self.wm_imgs.shape[0]
        for i in range(n):
            img = image.load_img(self.wm_imgs[i])
            wm_img.append(np.array(img._data_cache))
        wm_imgs = np.zeros([n]+list(wm_img[0].shape))
        for i in range(n):
            wm_imgs[i, ...] = wm_img[i]

        self.gm_imgs_3D = gm_imgs
        self.wm_imgs_3D = wm_imgs

    def get_3D_ROI(self, selectors):
        Atlas = ATLAS()
        gm_imgs_3D = self.gm_imgs_3D 
        wm_imgs_3D = self.wm_imgs_3D
        assert gm_imgs_3D.shape[0] == wm_imgs_3D.shape[0]
        n = gm_imgs_3D.shape[0]

        if self.ROIs_3D_gm is None:
            ROIs_3D_gm = []
            ROIs_3D_wm = []
        else:
            ROIs_3D_gm = self.ROIs_3D_gm
            ROIs_3D_wm = self.ROIs_3D_wm

        for selector in selectors:
            mask = Atlas.Mask_3D(selector)
            x0, x1, y0, y1, z0, z1 = DATA.bbox_3D(mask)
            x0 -= 2; y0 -=2; z0 -= 2
            x1 += 3; y1 +=3; z1 += 3
            mask_new = mask[x0:x1, y0:y1, z0:z1]
            
            ROI_3D_gm = np.zeros([n]+list(mask_new.shape))
            ROI_3D_wm = np.zeros([n]+list(mask_new.shape))
            for i in range(n):
                ROI_3D_gm[i, ...] = gm_imgs_3D[i, x0:x1, y0:y1, z0:z1] * mask_new
                ROI_3D_wm[i, ...] = wm_imgs_3D[i, x0:x1, y0:y1, z0:z1] * mask_new

            #f = plt.figure()
            #ax = f.add_subplot(111)
            #ax.imshow(ROI_3D_gm[20, :,:,15])

            #f1 = plt.figure()
            #ax1 = f1.add_subplot(111)
            #ax1.imshow(gm_imgs_3D[20, :,:,z0+15]*mask[:,:, z0+15])
            #plt.show()
                
            ROIs_3D_gm.append(ROI_3D_gm[..., np.newaxis])
            ROIs_3D_wm.append(ROI_3D_wm[..., np.newaxis])

        self.ROIs_3D_gm = ROIs_3D_gm
        self.ROIs_3D_wm = ROIs_3D_wm

    @staticmethod
    def bbox_3D(img):
        r = np.any(img, axis=(1, 2))
        c = np.any(img, axis=(0, 2))
        z = np.any(img, axis=(0, 1))

        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]

        return rmin, rmax, cmin, cmax, zmin, zmax
        
