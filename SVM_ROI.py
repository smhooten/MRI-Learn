import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import cross_validate

from DataPrep import DATA
from ATLAS import ATLAS

Atlas = ATLAS()

FileDir = os.getcwd()+'/SVM_Outputs/'
file_object = open(FileDir+'SVM_Log.txt', "a")
for i in range(3,46):
    ROI = Atlas.dataset_files.labels[i]
    FigName = "SVM_"+str(i)+"_"+ROI+".png"

    Data = DATA()
    Data.Train_Test(0.8)
    selectors = [i] #brain" #Hippo mask [34,35], whole brain "brain"
    Data.Add_MRI(selectors)
    Data.Split_Data()

    C_space = np.logspace(-4,1,50)
    train_score = []
    test_score = []
    print("Feature Size = "+str(Data.features_train.shape[1]))
    print("Started Training for "+FigName+"....")
    for C in C_space:
        SVM = svm.SVC(kernel='linear', C=C)
        cvs = cross_validate(SVM,Data.features_train,Data.labels_train, cv=4,return_train_score=True)
        train_score.append(np.mean(cvs["train_score"]))
        test_score.append(np.mean(cvs["test_score"]))
        # print("C="+str(C)+"Train"+str(np.mean(cvs["train_score"]))+"Valid"+str(np.mean(cvs["test_score"])))

    maxC = C_space[test_score.index(max(test_score))]

    print("Max Acc = "+str(np.around(max(test_score),4))+", with C = "+str(maxC))
    plt.figure()
    plt.semilogx(C_space,train_score)
    plt.semilogx(C_space,test_score)
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.legend(["Training","Validation"])
    plt.savefig(FileDir+FigName)


    print("Test Score Training.... ")
    SVM = svm.SVC(kernel='linear', C=maxC)
    SVM.fit(Data.features_train,Data.labels_train)
    pred_test=SVM.predict(Data.features_test)
    print("Test Accuracy = "+str(accuracy_score(pred_test,Data.labels_test)))

    file_object.write(str(i)+','+str(Data.features_train.shape[1])+','+str(maxC)+','+str(max(test_score))+','+str(accuracy_score(pred_test,Data.labels_test))+'\n')
