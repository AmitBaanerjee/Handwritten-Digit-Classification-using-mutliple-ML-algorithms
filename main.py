import pickle
import gzip
from PIL import Image
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
f.close()

validation_data_split = 0.3
num_epochs = 1000
model_batch_size = 128
tb_batch_size = 64
early_patience = 200
drop_out = 0.2
second_dense_layer_nodes  = 9

USPSMat  = []
USPSTar  = []
curPath  = 'USPSdata/Numerals'
savedImg = []
W = np.zeros((784,10))
W = np.transpose(W)
learningRate = 0.00001

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


# Seperating X & y values from the dataset
MNIST_traindata = training_data[0]
MNIST_traintarget = training_data[1]
MNIST_valdata = validation_data[0]
MNIST_valtarget = validation_data[1]
MNIST_testdata = test_data[0]
MNIST_testtarget = test_data[1]

# GetOneHotEncode() is used to convert each target label to One Hot Vectors
def GetOneHotEncode(target):
    onedata = []
    One = [0,0,0,0,0,0,0,0,0,0]
    for i in target:
        One[i] = 1
        onedata.append(One)
        One = [0,0,0,0,0,0,0,0,0,0]
    return onedata

MNIST_trainone = GetOneHotEncode(MNIST_traintarget)
MNIST_valone = GetOneHotEncode(MNIST_valtarget)
MNIST_testone = GetOneHotEncode(MNIST_testtarget)

# GetSoftMax() is used to compute SoftMax value for one sample of data
def GetSoftMax(data):
    unit = np.exp( np.dot(W,data))
    op = unit/np.sum(unit)
    return op

# GetLossFunction() is used to calculate Cross Entropy Error between the output Y & target t
def GetLossFunction(data,target):
    Total = []
    for i in data:
        SM = GetSoftMax(i)
        Total.append(SM)

    Er = -np.multiply(np.log(Total),target)
    Loss = np.sum(Er)
    return Loss, Total

# GetGradient() is used to perform Gradient of the Error value
def GetGradient(data, output, target):
  error = np.subtract(output,target)
  data_trans =  np.transpose(data)
  return np.dot(data_trans,error)

# GetPredictedOutput() is used to predict output values Y for Logistic Regression
def GetPredictedOutput(datarow):
  p = GetSoftMax(datarow)
  max = 0
  index = -1
  for i in range(10):
    if(max< p[i]):
      max = p[i]
      index = i
  return index

# GetAccuracy() is used to calculate cumulative accuracy based on the confusion matrix provided
def GetAccuracy(CMat):
    ConfSum = np.sum(CMat,axis=0)
    Num = []
    for i in range(len(CMat)):
        for j in range(len(CMat[i])):
            if(i==j):
                Num.append(CMat[i][j])
    Value = 0
    for i in range(len(ConfSum)):
        Value += Num[i]/ConfSum[i]
    Value = Value/len(Num)
    return float(Value)*100


# GetLogisticRegression() is used to call all the subordinate functions to train & test the Logistic Regression model
def GetLogisticRegression():
   hist = []

   print ('----------------------------------------------------')
   print ('--------------Please Wait for a minute!--------------')
   print ('----------------------------------------------------')
   global W
   for i in range(200):
       TrainLossE, TrainTotalSM= GetLossFunction(MNIST_traindata,MNIST_trainone)
       ValLossE, ValTotalSM= GetLossFunction(MNIST_valdata,MNIST_valone)
       DeltaE = GetGradient(MNIST_traindata, TrainTotalSM, MNIST_trainone)
       W = W - np.multiply(np.transpose(DeltaE),learningRate)
       hist.append([W,ValLossE])

   hist = np.asarray(hist)
   W = [hist[x,0] for x in range(np.shape(hist)[0]) if hist[x,1]==min(hist[:,1])]
   W = W[0]

   MNIST_Y = [GetPredictedOutput(d) for d in MNIST_testdata]
   USPS_Y = [GetPredictedOutput(d) for d in USPSMat]

   MNIST_ConfMat = confusion_matrix(MNIST_Y, MNIST_testtarget)
   USPS_ConfMat = confusion_matrix(USPS_Y, USPSTar)

   MNIST_Accuracy = GetAccuracy(MNIST_ConfMat)
   USPS_Accuracy = GetAccuracy(USPS_ConfMat)
   print ('---------Logistic Regression---------')
   print("Accuracy for MNIST Dataset:"+ str(MNIST_Accuracy))
   print("Accuracy for USPS Dataset:"+ str(USPS_Accuracy))

   plt.figure()
   plt.plot(hist[:,1])
   plt.xlabel('Number of Iterations')
   plt.ylabel('Validation Loss')

   return MNIST_Y, MNIST_Accuracy, USPS_Y, USPS_Accuracy, MNIST_ConfMat, USPS_ConfMat

# GetNeuralNetwork() is used to train & test MNIST & USPS dataset on MultiLayer Perceptron Neural Network model
def GetNeuralNetwork():

    mlp = MLPClassifier(hidden_layer_sizes=(50,),activation = 'relu', max_iter=100, alpha=1e-4,
                    solver='sgd', verbose=True, tol=1e-4, random_state=1,
                    learning_rate_init=0.1)

    mlp.fit(MNIST_traindata, MNIST_traintarget)

    MNIST_Y = mlp.predict(MNIST_testdata)
    USPS_Y = mlp.predict(USPSMat)

    MNIST_ConfMat = confusion_matrix(MNIST_Y, MNIST_testtarget)
    USPS_ConfMat = confusion_matrix(USPS_Y, USPSTar)

    MNIST_Accuracy = GetAccuracy(MNIST_ConfMat)
    USPS_Accuracy = GetAccuracy(USPS_ConfMat)

    print ('---------Neural Network---------')
    print('---------For MNIST Dataset---------')
    print("Training set score: %f" % mlp.score(MNIST_traindata, MNIST_traintarget))
    print("Validation set score: %f" % mlp.score(MNIST_valdata, MNIST_valtarget))
    print("Test set score: %f" % MNIST_Accuracy)
    print('---------For USPS Dataset---------')
    print("Test set score: %f" % USPS_Accuracy)

    plt.figure()
    plt.plot(mlp.loss_curve_)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Validation Loss')

    return MNIST_Y, MNIST_Accuracy, USPS_Y, USPS_Accuracy, MNIST_ConfMat,USPS_ConfMat

# GetSupportVectorMachine() is used to train & test MNIST & USPS dataset on Support Vector Machine model
def GetSupportVectorMachine():
    classifier = svm.SVC(C=200,kernel='rbf',gamma=0.01,cache_size=8000,probability=False, max_iter=-1)

    classifier.fit(MNIST_traindata, MNIST_traintarget)

    MNIST_Y = classifier.predict(MNIST_testdata)
    USPS_Y = classifier.predict(USPSMat)

    MNIST_ConfMat = confusion_matrix(MNIST_Y, MNIST_testtarget)
    USPS_ConfMat = confusion_matrix(USPS_Y, USPSTar)

    MNIST_Accuracy = GetAccuracy(MNIST_ConfMat)
    USPS_Accuracy = GetAccuracy(USPS_ConfMat)

    print ('---------Support Vector Machine---------')
    print('---------For MNIST Dataset---------')
    print("Test set score: %f" % MNIST_Accuracy)
    print('---------For USPS Dataset---------')
    print("Test set score: %f" % USPS_Accuracy)

    return MNIST_Y, MNIST_Accuracy, USPS_Y, USPS_Accuracy, MNIST_ConfMat,USPS_ConfMat

# GetSupportVectorMachine() is used to train & test MNIST & USPS dataset on Random Forest model
def GetRandomForest():
    classifier = RandomForestClassifier(n_estimators=100, max_depth=15)
    classifier.fit(MNIST_traindata, MNIST_traintarget)

    MNIST_Y = classifier.predict(MNIST_testdata)
    USPS_Y = classifier.predict(USPSMat)

    MNIST_ConfMat = confusion_matrix(MNIST_Y, MNIST_testtarget)
    USPS_ConfMat = confusion_matrix(USPS_Y, USPSTar)

    MNIST_Accuracy = GetAccuracy(MNIST_ConfMat)
    USPS_Accuracy = GetAccuracy(USPS_ConfMat)


    print ('---------Random Forest---------')
    print('---------For MNIST Dataset---------')
    print("Test set score: %f" % MNIST_Accuracy)
    print('---------For USPS Dataset---------')
    print("Test set score: %f" % USPS_Accuracy)

    return MNIST_Y, MNIST_Accuracy, USPS_Y, USPS_Accuracy, MNIST_ConfMat,USPS_ConfMat

# GetVotingClassifier() is used to provide a more balanced & accurate prediction of results based on the predicted outputs of other classifer models
def GetVotingClassifier(LP,LA,NP,NA,VP,VA,FP,FA,what):
    TotalPredict = []
    for i in range(len(LP)):
        TotalPredict.append([LP[i],NP[i],VP[i],FP[i]])

    TotalAccuracy = [LA,NA,VA,FA]

    VoteY = []

    for i in range(len(LP)):
        VoteForOne = np.zeros([10,1])
        for j in range(4):
            VoteForOne[TotalPredict[i][j]]+=TotalAccuracy[j] # Adding accuracy based on predicted Y
        FinalVote = np.argmax(VoteForOne) # Returns index of maximum row
        VoteY.append(FinalVote)

    if(what=="MNIST DATASET"):
        TestTarget = MNIST_testtarget
    else:
        TestTarget = USPSTar

    VoteConfMat = confusion_matrix(VoteY, TestTarget)
    VoteAccuracy = GetAccuracy(VoteConfMat)
    print("Test set score for Voting Classifier on "+what+": %f" % VoteAccuracy)

    return VoteConfMat

def main():
    print ('UBITname      = ameyakir')
    print ('Person Number = 50292574')
    print("--------Project 3: Handwritten Digit Classification---------")

    MNISTLogP, MNISTLogAccuracy, USPSLogP, USPSLogAccuracy , MNISTLogMat, USPSLogMat = GetLogisticRegression() # Running Logistic Regression Model
    MNISTNetP, MNISTNetAccuracy, USPSNetP, USPSNetAccuracy, MNISTNetMat, USPSNetMat = GetNeuralNetwork() # Running MultiLayer Perceptron Neural Network Model
    MNISTVectP, MNISTVectAccuracy, USPSVectP, USPSVectAccuracy,MNISTVectMat, USPSVectMat  = GetSupportVectorMachine() # Running SVM Model
    MNISTForestP, MNISTForestAccuracy, USPSForestP, USPSForestAccuracy, MNISTForestMat, USPSForestMat = GetRandomForest() # Running Random Forest Model

    MNISTVoteConfMat = GetVotingClassifier(MNISTLogP,MNISTLogAccuracy,MNISTNetP,MNISTNetAccuracy,MNISTVectP,MNISTVectAccuracy,MNISTForestP,MNISTForestAccuracy,"MNIST DATASET") # Calculating Majority Vote for MNIST Dataset
    USPSVoteConfMat = GetVotingClassifier(USPSLogP,USPSLogAccuracy,USPSNetP,USPSNetAccuracy,USPSVectP,USPSVectAccuracy,USPSForestP,USPSForestAccuracy,"USPS DATASET") # Calculating Majority Vote for USPS Dataset

    print('Confusion Matrix for Majority Voting Classifier on MNIST Dataset:')
    print(MNISTVoteConfMat)
    print('Confusion Matrix for Majority Voting Classifier on USPS Dataset:')
    print(USPSVoteConfMat)

if __name__ == '__main__':
    main()
