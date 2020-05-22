import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import cross_validate

from DataPrep import DATA
from ATLAS import ATLAS

from sklearn.decomposition import PCA

Atlas = ATLAS()

FileDir = os.getcwd()+'/PCA/'
FigName = "Brain_PCA"
file_object = open(FileDir+'PCA_Log.txt', "a")

Data = DATA()
Data.Train_Test(0.8,1234)
Data.Add_MRI("brain")
Data.Split_Data()

PCA_space = np.arange(1,30)*3
Train_Best = []
Valid_Best = []
Test_Best = []
MaxCs = []

for comp in PCA_space:
    pca = PCA(n_components=comp)
    pca.fit(Data.features_train[:,5:])

    X_train_pca = pca.transform(Data.features_train[:,5:])
    X_test_pca = pca.transform(Data.features_test[:,5:])

    train = np.hstack((X_train_pca,Data.features_train[:,0:5]))
    test = np.hstack((X_test_pca,Data.features_test[:,0:5]))

    print("Training....")
    train_score = []
    test_score = []
    C_space = np.logspace(-7,3,30)
    for C in C_space:
        SVM = svm.SVC(kernel='linear', C=C)
        cvs = cross_validate(SVM,train,Data.labels_train, cv=4,return_train_score=True)
        train_score.append(np.mean(cvs["train_score"]))
        test_score.append(np.mean(cvs["test_score"]))
    #     print("C="+str(C)+"Train"+str(np.mean(cvs["train_score"]))+"Valid"+str(np.mean(cvs["test_score"])))

    maxC = C_space[test_score.index(max(test_score))]

    SVM = svm.SVC(kernel='linear', C=maxC)
    SVM.fit(train,Data.labels_train)
    pred_test=SVM.predict(test)

    Train_Best.append(train_score[test_score.index(max(test_score))])
    Valid_Best.append(max(test_score))
    Test_Best.append(accuracy_score(pred_test,Data.labels_test))
    MaxCs.append(maxC)

    print("PCA = "+str(comp))
    print("Max Acc = "+str(np.around(max(test_score),4))+", with C = "+str(maxC)+" Test Acc = "+str(accuracy_score(pred_test,Data.labels_test)))
    file_object.write(str(comp)+','+str(maxC)+','+str(max(test_score))+','+str(accuracy_score(pred_test,Data.labels_test))+'\n')

plt.plot(PCA_space,Train_Best)
plt.plot(PCA_space,Valid_Best)
plt.plot(PCA_space,Test_Best)
plt.xlabel("PCA components")
plt.ylabel("Accuracy")
plt.legend(["Train","Valid","Test"])
plt.savefig(FileDir+FigName)
