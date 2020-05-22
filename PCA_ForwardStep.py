import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn import svm,neighbors
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate,KFold

from DataPrep import DATA
from ATLAS import ATLAS

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.cross_decomposition import CCA


def Train_Test_SVM(train,test):
    train_score=[]
    valid_score=[]
    std_valid=[]
    C_space = np.logspace(-7,0,25)
    for C in C_space:
        SVM = svm.SVC(kernel='linear',C=C)
        cvs = cross_validate(SVM,train,Data.labels_train, cv=4,return_train_score=True)
        train_score.append(np.mean(cvs["train_score"]))
        valid_score.append(np.mean(cvs["test_score"]))
        std_valid.append(np.std(cvs["test_score"]))

    max_idx = valid_score.index(max(valid_score))
    maxC = C_space[max_idx]

    SVM = svm.SVC(kernel='linear',C=maxC)
    SVM.fit(train,Data.labels_train)
    test_acc = SVM.score(test,Data.labels_test)

    return(maxC,train_score[max_idx],valid_score[max_idx],std_valid[max_idx],test_acc)

Atlas = ATLAS()

Data = DATA()
Data.Train_Test(0.8,12345)
Data.Add_MRI("brain")
Data.Split_Data()

comp = 50

pca = PCA(n_components=comp, whiten=False)
pca.fit(Data.features_train[:,5:])

X_train_pca = pca.transform(Data.features_train[:,5:])
X_test_pca = pca.transform(Data.features_test[:,5:])

train_pca = np.hstack((X_train_pca,Data.features_train[:,0:5]))
test_pca = np.hstack((X_test_pca,Data.features_test[:,0:5]))

#Most predictive features (5. 13. 47. 31. 45.)

idx_list = np.array([])
sweep = np.arange(0,train_pca.shape[1])
Train_Max = []
Valid_Max = []
Valid_STD = []
Test_Max = []
for i in range(30):
    Train_Acc = []
    Valid_Acc = []
    Valid_std = []
    Test_Acc = []
    for idx in sweep:
        idxs = np.hstack((idx_list,idx)).astype('int')
        if len(idxs)==1:
            train = np.reshape(train_pca[:,idxs],(-1,1))
            test = np.reshape(test_pca[:,idxs],(-1,1))
        else:
            train = train_pca[:,idxs]
            test = test_pca[:,idxs]
        SVM_PCA = Train_Test_SVM(train,test)
        Train_Acc.append(SVM_PCA[1])
        Valid_Acc.append(SVM_PCA[2])
        Valid_std.append(SVM_PCA[3])
        Test_Acc.append(SVM_PCA[4])

    Train_Acc = np.array(Train_Acc)
    Valid_Acc = np.array(Valid_Acc)
    idx_list = np.hstack((idx_list,sweep[Valid_Acc.argmax()]))
    sweep = np.delete(sweep, Valid_Acc.argmax())
    Train_Max.append(Train_Acc[Valid_Acc.argmax()])
    Valid_Max.append(max(Valid_Acc))
    Valid_STD.append(Valid_std[Valid_Acc.argmax()])
    Test_Max.append(Test_Acc[Valid_Acc.argmax()])
    print(idx_list)

xrange = np.arange(1,31)

plt.errorbar(x = xrange, y=Valid_Max,yerr=Valid_STD)
plt.plot(xrange,Test_Max,'.')
plt.plot(xrange,Train_Max,'--')
plt.xlabel("Components")
plt.ylabel("Accuracy")
plt.legend(["Test","Train (Avg)","Valid"],loc='lower right')
plt.savefig("ForwardStep.pdf")
