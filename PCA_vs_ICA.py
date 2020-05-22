import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import svm,neighbors
from sklearn.model_selection import cross_validate

from DataPrep import DATA
from ATLAS import ATLAS

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def Train_Test_SVM(train,test):
    print("Training SVM...")
    train_score=[]
    valid_score=[]
    std_valid=[]
    C_space = np.logspace(-4,15,30)
    for C in C_space:
        SVM = svm.LinearSVC(penalty='l2',C=C,max_iter=1000000,dual=False)
        cvs = cross_validate(SVM,train,Data.labels_train, cv=4,return_train_score=True)
        train_score.append(np.mean(cvs["train_score"]))
        valid_score.append(np.mean(cvs["test_score"]))
        std_valid.append(np.std(cvs["test_score"]))

    max_idx = valid_score.index(max(valid_score))
    maxC = C_space[max_idx]

    SVM = svm.LinearSVC(penalty='l2',C=maxC,max_iter=1000000,dual=False)
    SVM.fit(train,Data.labels_train)
    test_acc = SVM.score(test,Data.labels_test)

    return(maxC,train_score[max_idx],valid_score[max_idx],std_valid[max_idx],test_acc)

def Train_Test_ADA(train,test):
    print("Training ADA...")
    train_score=[]
    valid_score=[]
    std_valid=[]
    n_space = np.arange(1,20)*2
    for n in n_space:
        ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=n, random_state=0)
        cvs = cross_validate(ada,train,Data.labels_train, cv=4,return_train_score=True)
        train_score.append(np.mean(cvs["train_score"]))
        valid_score.append(np.mean(cvs["test_score"]))
        std_valid.append(np.std(cvs["test_score"]))

    max_idx = valid_score.index(max(valid_score))
    maxN = n_space[max_idx]

    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=n, random_state=0)
    ada.fit(train,Data.labels_train)
    test_acc = ada.score(test,Data.labels_test)

    return(maxN,train_score[max_idx],valid_score[max_idx],std_valid[max_idx],test_acc)

def Train_Test_NN(train,test):
    print("Training NN...")
    train_score=[]
    valid_score=[]
    std_valid=[]
    neighbor_space=np.arange(1,12)
    for neigh in neighbor_space:
        NN = neighbors.KNeighborsClassifier(neigh)
        cvs = cross_validate(NN,train,Data.labels_train, cv=4,return_train_score=True)
        train_score.append(np.mean(cvs["train_score"]))
        valid_score.append(np.mean(cvs["test_score"]))
        std_valid.append(np.std(cvs["test_score"]))

    max_idx = valid_score.index(max(valid_score))
    maxN = neighbor_space[max_idx]

    NN = neighbors.KNeighborsClassifier(maxN)
    NN.fit(train,Data.labels_train)
    test_acc = NN.score(test,Data.labels_test)

    return(maxN,train_score[max_idx],valid_score[max_idx],std_valid[max_idx],test_acc)

def PrintFile(FILE,comp,data):
    FILE.write(str(comp)+',')
    for dat in data:
        FILE.write(str(dat)+',')
    FILE.write('\n')

Atlas = ATLAS()

FileDir = os.getcwd()+'/PCA_vs_ICA/'
FigName = "Brain_PCA"
file_SVM_p = open(FileDir+'SVM_PCA_Log.txt', "a")
file_SVM_i = open(FileDir+'SVM_ICA_Log.txt', "a")

file_ADA_p = open(FileDir+'ADA_PCA_Log.txt', "a")
file_ADA_i = open(FileDir+'ADA_ICA_Log.txt', "a")

file_NN_p = open(FileDir+'NN_PCA_Log.txt', "a")
file_NN_i = open(FileDir+'NN_ICA_Log.txt', "a")

Data = DATA()
Data.Train_Test(0.8,12345)
Data.Add_MRI("brain")
Data.Split_Data()

Comp_space = np.arange(1,60)

SVM_PCA_Valid = []
SVM_PCA_Valid_STD = []
SVM_PCA_Test = []

SVM_ICA_Valid = []
SVM_ICA_Valid_STD = []
SVM_ICA_Test = []

ADA_PCA_Valid = []
ADA_PCA_Valid_STD = []
ADA_PCA_Test = []

ADA_ICA_Valid = []
ADA_ICA_Valid_STD = []
ADA_ICA_Test = []

NN_PCA_Valid = []
NN_PCA_Valid_STD = []
NN_PCA_Test = []

NN_ICA_Valid = []
NN_ICA_Valid_STD = []
NN_ICA_Test = []

for comp in Comp_space:
    pca = PCA(n_components=comp, whiten=False)
    ica = FastICA(n_components=comp, whiten=True,max_iter=1000000)

    print("PCA Fit...")
    pca.fit(Data.features_train[:,5:])
    print("ICA Fit...")
    ica.fit(Data.features_train[:,5:])

    X_train_pca = pca.transform(Data.features_train[:,5:])
    X_test_pca = pca.transform(Data.features_test[:,5:])

    train_pca = np.hstack((X_train_pca,Data.features_train[:,0:5]))
    test_pca = np.hstack((X_test_pca,Data.features_test[:,0:5]))

    X_train_ica = ica.transform(Data.features_train[:,5:])
    X_test_ica = ica.transform(Data.features_test[:,5:])

    train_ica = np.hstack((X_train_ica,Data.features_train[:,0:5]))
    test_ica = np.hstack((X_test_ica,Data.features_test[:,0:5]))

    print("Components = "+str(comp))

    SVM_PCA = Train_Test_SVM(train_pca,test_pca)
    SVM_ICA = Train_Test_SVM(train_ica,test_ica)

    print("SVM (PCA/ICA)")
    print(SVM_PCA)
    print(SVM_ICA)

    ADA_PCA = Train_Test_ADA(train_pca,test_pca)
    ADA_ICA = Train_Test_ADA(train_ica,test_ica)

    print("ADA (PCA/ICA)")
    print(ADA_PCA)
    print(ADA_ICA)

    NN_PCA = Train_Test_NN(train_pca,test_pca)
    NN_ICA = Train_Test_NN(train_pca,test_pca)

    print("NN (PCA/ICA)")
    print(NN_PCA)
    print(NN_ICA)
    PrintFile(file_SVM_p,comp,SVM_PCA)
    PrintFile(file_SVM_i,comp,SVM_ICA)

    PrintFile(file_ADA_p,comp,ADA_PCA)
    PrintFile(file_ADA_i,comp,ADA_ICA)

    PrintFile(file_NN_p,comp,NN_PCA)
    PrintFile(file_NN_i,comp,NN_ICA)

    SVM_PCA_Valid.append(SVM_PCA[2])
    SVM_PCA_Valid_STD.append(SVM_PCA[3])
    SVM_PCA_Test.append(SVM_PCA[4])

    SVM_ICA_Valid.append(SVM_ICA[2])
    SVM_ICA_Valid_STD.append(SVM_ICA[3])
    SVM_ICA_Test.append(SVM_ICA[4])

    ADA_PCA_Valid.append(ADA_PCA[2])
    ADA_PCA_Valid_STD.append(ADA_PCA[3])
    ADA_PCA_Test.append(ADA_PCA[4])

    ADA_ICA_Valid.append(ADA_ICA[2])
    ADA_ICA_Valid_STD.append(ADA_ICA[3])
    ADA_ICA_Test.append(ADA_ICA[4])

    NN_PCA_Valid.append(NN_PCA[2])
    NN_PCA_Valid_STD.append(NN_PCA[3])
    NN_PCA_Test.append(NN_PCA[4])

    NN_ICA_Valid.append(NN_ICA[2])
    NN_ICA_Valid_STD.append(NN_ICA[3])
    NN_ICA_Test.append(NN_ICA[4])

plt.errorbar(Comp_space,SVM_PCA_Valid,yerr=SVM_PCA_Valid_STD)
plt.plot(Comp_space,SVM_PCA_Test,'.')
plt.xlabel("PCA components")
plt.ylabel("Accuracy")
plt.legend(["Test","Valid"])
plt.savefig(FileDir+'SVM_PCA.png')

plt.figure()
plt.errorbar(Comp_space,SVM_ICA_Valid,yerr=SVM_ICA_Valid_STD)
plt.plot(Comp_space,SVM_ICA_Test,'.')
plt.xlabel("ICA components")
plt.ylabel("Accuracy")
plt.legend(["Test","Valid"])
plt.savefig(FileDir+'SVM_ICA.png')

plt.figure()
plt.errorbar(Comp_space,ADA_PCA_Valid,yerr=ADA_PCA_Valid_STD)
plt.plot(Comp_space,ADA_PCA_Test,'.')
plt.xlabel("PCA components")
plt.ylabel("Accuracy")
plt.legend(["Test","Valid"])
plt.savefig(FileDir+'ADA_PCA.png')

plt.figure()
plt.errorbar(Comp_space,ADA_ICA_Valid,yerr=ADA_ICA_Valid_STD)
plt.plot(Comp_space,ADA_ICA_Test,'.')
plt.xlabel("ICA components")
plt.ylabel("Accuracy")
plt.legend(["Test","Valid"])
plt.savefig(FileDir+'ADA_ICA.png')

plt.figure()
plt.errorbar(Comp_space,NN_PCA_Valid,yerr=NN_PCA_Valid_STD)
plt.plot(Comp_space,NN_PCA_Test,'.')
plt.xlabel("PCA components")
plt.ylabel("Accuracy")
plt.legend(["Test","Valid"])
plt.savefig(FileDir+'NN_PCA.png')

plt.figure()
plt.errorbar(Comp_space,NN_ICA_Valid,yerr=NN_ICA_Valid_STD)
plt.plot(Comp_space,NN_ICA_Test,'.')
plt.xlabel("ICA components")
plt.ylabel("Accuracy")
plt.legend(["Test","Valid"])
plt.savefig(FileDir+'NN_ICA.png')
