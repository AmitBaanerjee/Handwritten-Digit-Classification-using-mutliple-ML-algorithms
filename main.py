
print("--------Project 3:Classificatio------")
print("--------UBName :Amit Banerjee--------")
print("--------UBID : 50287084--------")

#importing necessary libraries 
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder

#extracting MNIST data
filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
f.close()

#extracting USPS data
from PIL import Image
import os

USPSMat  = []
USPSTar  = []
curPath  = 'USPSdata/Numerals'
savedImg = []

for j in range(0,10):
    curFolderPath = curPath + '/' + str(j)
    imgs =  os.listdir(curFolderPath)
    for img in imgs:
        curImg = curFolderPath + '/' + img
        if curImg[-3:] == 'png':
            img = Image.open(curImg,'r')
            img = img.resize((28, 28))
            savedImg = img
            imgdata = (255-np.array(img.getdata()))/255
            USPSMat.append(imgdata)
            USPSTar.append(j)

usps_data=np.asarray(USPSMat)

#Pre-processing MNIST data
m_training_target = training_data[1]
m_training_data = training_data[0]
m_test_target = test_data[1]
m_test_data = test_data[0]
m_validation_target = validation_data[1]
m_validation_data = validation_data[0]

#Encoding MNIST target data for applying softmax function
def onehot(target):
    hot_x=OneHotEncoder(categorical_features=[0],dtype=np.int) 
    mt=target.reshape(-1,1)
    target=hot_x.fit_transform(mt).toarray()
    return target

m_training_target=onehot(m_training_target)
m_validation_target=onehot(m_validation_target)
m_test_target=onehot(m_test_target)

m_training_target=m_training_target.tolist()
m_test_target=m_test_target.tolist()
m_validation_target=m_validation_target.tolist()

#Initializing weight vector for gradient descent
w=np.zeros([m_training_data.shape[1],len(m_training_target[0])],dtype=int)

#Function to calculate softmax probabilities
def softmax(data):
    num=np.dot(np.transpose(w),data)
    num=np.exp(num)
    den=np.sum(num)
    yk=num/den
    return yk

#Loss function defination for logistic regression
def lossfunction(data,target):
    finalsoft=[]
    for i in data:
        val=softmax(i)
        finalsoft.append(val)
    loss=-np.multiply((target),(np.log(finalsoft)))
    loss=np.sum(loss)
    return loss,finalsoft
#Function for calculating gradient in SGD
def gradient(data, yk, target):
  loss = np.subtract(yk,target)
  dataT =  np.transpose(data)
  return np.matmul(dataT,loss)

#Function to return final class for input data row by selecting class with maximum probability
def predict(datarow):
  p = softmax(datarow)
  max = 0 
  index = -1
  for i in range(10):
    if(max< p[i]):
      max = p[i]
      index = i
  return index

#Function to calculate accuracy of model using confusion matrix as input
def accuracy(c):
    accval=0
    acclist=[]
    for i in range(len(c)):
        s=0.01
        for j in range(len(c[i])):
            s=s+c[i][j]
            if(i==j):
                num=c[i][j]
        accval=(num/s)*100
        acclist.append(accval)
    totalacc=sum(acclist)/len(acclist)
    return totalacc,acclist

#Function to perform logistic regression on input data using gradient descent optimizer
def logistic(data, target):
  global w
  learning_rate =  0.0001
  record = []
  for i in range(300):
    loss,yk=lossfunction(data,target)
    loss_val,yk_val=lossfunction(m_validation_data,m_validation_target)
    grad=gradient(data,yk,target)
    w = w - learning_rate*grad
    record.append([w,loss])
  record = np.array(record)
  w = [record[x,0] for x in range(np.shape(record)[0]) if record[x,1]==min(record[:,1])]
  w = w[0]
  pre_test= [predict(d) for d in m_test_data] 
  test_target_confusion=test_data[1]
  c=confusion_matrix(test_target_confusion, pre_test)
  acc,lomlist=accuracy(c)
  #USPS Dataset
  pre_testu= [predict(d) for d in usps_data] 
  cu=confusion_matrix(USPSTar, pre_testu)
  accu,loulist=accuracy(cu)
  print("Accuracy of USPS Logistic Regression implementation :- ",accu)
  return acc,accu,pre_test,pre_testu,lomlist,loulist,c,cu

a_logistic,a_ulogistic,logistic_ypred,logistic_ypredu,lomlist,loulist,lomc,louc=logistic(m_training_data, m_training_target)
print("Accuracy of MNIST Logistic Regression implementation:- ",a_logistic)

'************* NEURAL NETWORK implementation *****************'
from sklearn.neural_network import MLPClassifier
import pickle
import gzip
import numpy as np

filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
f.close()

m_training_target = training_data[1]
m_training_data = training_data[0]
m_test_target = test_data[1]
m_test_data = test_data[0]
m_validation_target = validation_data[1]
m_validation_data = validation_data[0]

#setting hyper parameters for NN training model
mlp = MLPClassifier(hidden_layer_sizes=(100,25), max_iter=100, alpha=1e-4,
                    solver='sgd', verbose=False, tol=1e-4, random_state=1,
                    learning_rate_init=0.01,batch_size=1000)

mlp.fit(m_training_data, m_training_target)
print("Training set score: %f" % mlp.score(m_training_data, m_training_target))
print("Test set score: %f" % mlp.score(m_test_data, m_test_target))
#calculating target predictions for MNIST dataset and get accuracy
NN_ypred= mlp.predict(m_test_data)
nnmc=confusion_matrix(m_test_target, NN_ypred)
a_NN,nnmlist=accuracy(nnmc)
print('Accuracy of MNIST Neural Network implementation :- ',a_NN)
#calculating target predictions for USPS dataset and accuracy
predu= mlp.predict(USPSMat)
nnuc=confusion_matrix(USPSTar, predu)
nnua,nnulist=accuracy(nnuc)
print('Accuracy of USPS Neural Network implementation :- ',nnua)

'**************** SVM Implementation*****************'
from sklearn import svm
#function to set hyper parameters and train SVM model for MNIST and USPS dataset
def GetSupportVectorMachine():
    classifier = svm.SVC(C=190,kernel='rbf',gamma=0.01,cache_size=8000,probability=False)
    classifier.fit(m_training_data, m_training_target)
    ypred=classifier.predict(m_test_data)
    c_svm=confusion_matrix(m_test_target,ypred)
    svm_acc,acclistsvm=accuracy(c_svm)
    print("Accuracy of MNIST SVM implementation :- ",svm_acc)
    
    #SVM for USPS dataset
    ypredusps=classifier.predict(USPSMat)
    c_svm_usps=confusion_matrix(USPSTar,ypredusps)
    svm_acc_usps,acclistsvmusps=accuracy(c_svm_usps)
    print("Accuracy of USPS SVM implementation :- ",svm_acc_usps)
    return ypred,ypredusps,svm_acc,svm_acc_usps,acclistsvm,acclistsvmusps,c_svm,c_svm_usps
SVM_ypred,SVM_ypredu,a_SVM,a_SVMu,acclistsvm,acclistsvmusps,svmmc,svmuc=GetSupportVectorMachine()
    
'**************RANDOM FOREST*****************'
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
#setting RF model parameters to train on MNIST dataset
classifier2 = RandomForestClassifier(n_estimators=400,min_samples_split=2);
classifier2.fit(m_training_data, m_training_target) 
ypred_rf=classifier2.predict(m_test_data)
c_rf=confusion_matrix(m_test_target,ypred_rf)
rf_acc,acclistrf=accuracy(c_rf)
print("Accuracy of MNIST Random Forest implementation :- ",rf_acc)
#generating prediction values for USPS dataset
ypred_rf_usps=classifier2.predict(USPSMat)
c_rf_usps=confusion_matrix(USPSTar,ypred_rf_usps)
rf_acc_usps,acclistrfusps=accuracy(c_rf_usps)
print("Accuracy of USPS Random Forest implementation :- ",rf_acc_usps)

'**************COMBINATION OF CLASSIFIERS*************'
#combined classifier for MNIST dataset
#Storing target prediction values in list
cl = [np.asarray(logistic_ypred), NN_ypred, SVM_ypred, ypred_rf]
cl2 = np.transpose(cl)
#storing accuracy values of all classifiers in list
acc = [a_logistic, a_NN, a_SVM, rf_acc]

t =[]
for i in range(len(cl2)):
    #using an empty array to calculate final class for each datarow
    classes = np.zeros([10,1])
    for j in range(len(cl2[i])):
        classes[cl2[i][j]]+=acc[j]
    #appending final class into new list t
    t.append(np.argmax(classes))
c_combine=confusion_matrix(m_test_target,t)
a_combine,acclistcombine=accuracy(c_combine)
print("Accuracy of combined classifier on MNIST dataset :- ",a_combine)

#combined classifier for USPS dataset
clu = [np.asarray(logistic_ypredu), predu, SVM_ypredu, ypred_rf_usps]
cl2u = np.transpose(clu)
accu = [a_ulogistic, nnua, a_SVMu, rf_acc_usps]

tu =[]
for i in range(len(cl2u)):
    classesu = np.zeros([10,1])
    for j in range(len(cl2u[i])):
        classesu[cl2u[i][j]]+=accu[j]
    tu.append(np.argmax(classesu))
cu_combine=confusion_matrix(USPSTar,tu)
au_combine,acclistucombine=accuracy(cu_combine)
print("Accuracy of combined classifier on USPS dataset :- ",au_combine)

