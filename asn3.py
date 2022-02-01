from operator import concat, index, neg
from os import system, write
import random
import numpy
import sys
import copy
from numpy.core.records import array
#asn3.py
#Jared Matson
#1570490

#This program is a neural network to regongnize handwritten numbers
#Given a test set, will train itself to recognize up to 3 numbers (1,7,8 in the current training set)
#After training to an accuracy of 92%, the program will write the weights of each layer into two seperate files ("inputCorrectWeights.txt and hiddenCorrectWeights.txt") to be used for testing
#If the person using this program runs the test file, the constructed neural network will grab the weights from the files listed above, and attempt to classify each test input, writing the results into test_output.txt
#Each test and train input is 256 numbers, representing one tile in a 16x16 picture that makes up the handwritten image

#class Node
#This represents each specific node in the neural network
#Value -> The value of the node
#Weights -> A list of weights, a weight at index 0 for example will represent the weight value going to the 0th node
#Delta -> The delta value of the node during backpropogation
class Node:
    def __init__(self, value) -> None:
        self.value = value
        self.weights = []
        self.delta = 0 

#sigmoid()
#Given a value, will return its sigmoid activation function value
#Parameters: Value -> The value that you want to see the activation value of
#Returns -> sigmoid value
def sigmoid(value):
    return 1/(1+ numpy.exp(neg(value)))

#FPrime()
#Given a sigmoid value, will return its Fprime value that is used in back propogation
#Parameters: sigmoidValue -> Value that you want to Fprime
#Returns -> FPrime value
def FPrime(sigmoidValue):
    return(sigmoidValue*(1-sigmoidValue))

#initializeNetwork()
#Given terminal arguement, will construct the neural network basics, creating bias nodes and a unique node for each input
#Parameters: terminalArguement -> The terminal arguements inputted
def initializeNetwork(terminalArguement):
    inputLayer = []
    hiddenLayer = []
    outputLayer = []
    inputLayer.append(Node(-1)) #bias node for input layer
    hiddenLayer.append(Node(-1)) #bias node for hidden layer
    for inputNode in range(256): #number of input nodes
        inputLayer.append(Node(None))
    for hiddenNode in range(10): #number of hidden nodes
        hiddenLayer.append(Node(None))
    for outputNode in range(3): #number of output nodes
        outputLayer.append(Node(None)) 
    
    initializeWeights(inputLayer, hiddenLayer, outputLayer, terminalArguement)

#initializeWeights()
#Now that the neural network is initialized, we can assign initial weights to each node
#If the terminalArguement inputs test.txt, this method will search inputCorrectWeights and hiddenCorrectWeights and assign weights from there
#If the terminalArguement inputs train.txt, this method will assign random weights to each node in order to train it
#Parameters:
# inputLayer -> The input layer of the network
# hiddenLayer -> The hidden layer of the network
# outputLayer -> The outputLayer of the network
# terminalArguement -> the terminal arguement inputted
def initializeWeights(inputLayer, hiddenLayer, outputLayer, terminalArguement):
    if terminalArguement[1] == "test.txt": #we already tested, grab weights that had high accuracy
        inputWeightFile = open("D:/Programming/Repositories/BackpropogationDigitRecognition/BackpropogationDigitRecognition/inputCorrectWeights.txt","r")#desktop
        hiddenWeightFile = open("D:/Programming/Repositories/BackpropogationDigitRecognition/BackpropogationDigitRecognition/hiddenCorrectWeights.txt","r")#desktop
        #read inputLayer weights
        lines = inputWeightFile.readlines() 
        index = 0
        for line in lines:
            for word in line.split():
                inputLayer[index].weights.append(float(word))    
            index += 1
        #read hiddenLayer weights
        index = 0
        lines = hiddenWeightFile.readlines()
        for line in lines:
            for word in line.split():
                hiddenLayer[index].weights.append(float(word))
            index += 1
        testNetwork(inputLayer, hiddenLayer, outputLayer, terminalArguement)
      
    else:   #we have not yet constructed the network, set random weights then train
        for node in inputLayer:
            for i in range(10):
                node.weights.append(round(random.uniform(-1,1),5)) #before it was 0,0.1
        for node in hiddenLayer:
            for i in range(3):
                node.weights.append(round(random.uniform(-1,1),5))  #before it was 0,0.05
        trainNetwork(inputLayer, hiddenLayer, outputLayer, terminalArguement, 0)
#writeToFile()
#During the testing phase, this method will write the actual output of the neural network as opposed to its expected output
#Parameters:
#expectedOutputIndex -> The index of the correct output
#ActualOutputIndex -> The index of the output of the training network for the iteration
def writeToFile(expectedOutputIndex, ActualOutputIndex):
    file = open("D:/Programming/Repositories/BackpropogationDigitRecognition/BackpropogationDigitRecognition/train_output.txt","a")#desktop
    #file = open("D:/Programming/Repo/backPropogationDigitRecognition/BackpropogationDigitRecognition/train_output.txt", "a") #laptop
    #writing the training algorithms output
    if(ActualOutputIndex == 0):
        file.write("1")
    elif(ActualOutputIndex == 1):
        file.write("8")
    elif(ActualOutputIndex == 2):
        file.write("9")
    #writing the expected output
    if(expectedOutputIndex == 0):
        file.write("                           1\n")
    elif(expectedOutputIndex == 1):
        file.write("                           8\n")
    elif(expectedOutputIndex == 2):
        file.write("                           9\n")
    file.close() 

#trainNetwork()
#This method controls the training process of the network, ensuring that it trains for another epoch if the current weights 
#do not reach an accuracy of 92%+
#Parameters:
# inputLayer -> The input layer of the network
# hiddenLayer -> The hidden layer of the network
# outputLayer -> The outputLayer of the network
# terminalArguement -> the terminal arguement inputted
# epochNumber -> The current epoch the network is training on
def trainNetwork(inputLayer, hiddenLayer, outputLayer, terminalArguement, epochNumber):
    epochNumber += 1
    fileToWrite = open("D:/Programming/Repositories/BackpropogationDigitRecognition/BackpropogationDigitRecognition/"+terminalArguement[2],"w")#desktop
    #fileToWrite = open("D:/Programming/Repo/backPropogationDigitRecognition/BackpropogationDigitRecognition/train_output.txt", "w") #laptop
    file = open("D:/Programming/Repositories/BackpropogationDigitRecognition/BackpropogationDigitRecognition/"+terminalArguement[1],"r")#desktop
    #file = open("D:/Programming/Repo/backPropogationDigitRecognition/BackpropogationDigitRecognition/"+terminalArguement[0],"r") #laptop

    fileToWrite.write("my_predicted_digit   target(correct digit)\n")
    fileToWrite.close()
    answer = []
    actualOutput = -1
    expectedOutput = -1
    correctlyClassified = 0
    numberOfClassifications = 0
    lines = file.readlines()
    for line in lines:
        answer = []
        indexInput = 1
        numberOfClassifications += 1 #per input
        for word in line.split(): #for each square input in the 16x16 picture                                       
            if(len(word)) == 6: #making sure we dont include the last three numbers in each line
                inputLayer[indexInput].value = int(float(word))
                indexInput += 1
            else: ##getting test result
                answer.append(word)
                if len(answer) == 3: #values have been inputted
                    if answer == ['1','0','0']:
                        expectedOutput = 0 #index of what the highest probability should be
                    elif answer == ['0','1','0']:
                        expectedOutput = 1
                    elif answer == ['0','0','1']:
                        expectedOutput = 2
                    actualOutput = passForward(inputLayer, hiddenLayer, outputLayer)
                    writeToFile(expectedOutput, actualOutput.index(max(actualOutput))) ##start here by writing to file
                    if expectedOutput == actualOutput.index(max(actualOutput)):#the program correctly guessed the output, we want to find the index instead of the actual values to easier compare
                        correctlyClassified += 1
                        continue
                    else:
                        newValues = backPropogation(inputLayer, hiddenLayer, outputLayer, answer)

    print(correctlyClassified / numberOfClassifications) #for each epoch, print accuracy of the network
    if(correctlyClassified / numberOfClassifications < 0.92):
        trainNetwork(newValues[0], newValues[1], newValues[2], terminalArguement, epochNumber)
    else: #if the network is above 92%, we can write its weight values down
        #file = open("D:/Programming/Repo/backPropogationDigitRecognition/BackpropogationDigitRecognition/train_output.txt", "a") #laptop
        file = open("D:/Programming/Repositories/BackpropogationDigitRecognition/BackpropogationDigitRecognition/"+terminalArguement[2],"a")#desktop
        file.write("\n")
        file.write("Accuracy: " + str(correctlyClassified) +"/" +str(numberOfClassifications) +" = "+str(((correctlyClassified/numberOfClassifications) * 100))+"%")
        file.close()
        print("epoch number"+ str(epochNumber))
        writeWeights(newValues[0], newValues[1])

#writeWeights()
#After the testing process gets a accuracy of 92%, this method  will write the networks weights to a file to be used for testing
#Parameters:
# inputLayer -> The input layer of the network
# hiddenLayer -> The hidden layer of the network
def writeWeights(inputLayer, hiddenLayer):
    fileInput = open("D:/Programming/Repositories/BackpropogationDigitRecognition/BackpropogationDigitRecognition/inputCorrectWeights.txt","w")
    fileHidden = open("D:/Programming/Repositories/BackpropogationDigitRecognition/BackpropogationDigitRecognition/hiddenCorrectWeights.txt","w")

    for inputNode in inputLayer:
        for weight in inputNode.weights:
            fileInput.write(str(weight) +" ")
        fileInput.write("\n")
    fileInput.close()
    for hiddenNode in hiddenLayer:
        for weight in hiddenNode.weights:
            fileHidden.write(str(weight) + " ")
        fileHidden.write("\n")
    fileHidden.close()
    

#testNetwork()
#This method will read the weights that were successful in the training phase, and test itself from the test file given
#Parameters:
# inputLayer -> The input layer of the network
# hiddenLayer -> The hidden layer of the network
# outputLayer -> The outputLayer of the network
# terminalArguement -> the terminal arguement inputted
def testNetwork(inputLayer, hiddenLayer, outputLayer, terminalArguement):
    file = open("D:/Programming/Repositories/BackpropogationDigitRecognition/BackpropogationDigitRecognition/"+terminalArguement[1],"r")#desktop
    #file = open("D:/Programming/Repo/backPropogationDigitRecognition/BackpropogationDigitRecognition/"+terminalArguement,"r") #laptop
    fileToWrite = open("D:/Programming/Repositories/BackpropogationDigitRecognition/BackpropogationDigitRecognition/"+terminalArguement[2],"w")#desktop
    #fileToWrite = open("D:/Programming/Repo/backPropogationDigitRecognition/BackpropogationDigitRecognition/test_output.txt", "w") #laptop


    lines = file.readlines()
    for line in lines:#read test file
        indexInput = 1
        for word in line.split():
            inputLayer[indexInput].value = int(float(word))
            indexInput += 1
        actualOutput = passForward(inputLayer, hiddenLayer, outputLayer)
        numberToWrite = actualOutput.index(max(actualOutput))
        #write output to file
        if(numberToWrite == 0):
            fileToWrite.write("1\n")
        elif(numberToWrite == 1):
            fileToWrite.write("8\n")
        elif(numberToWrite == 2):
            fileToWrite.write("9\n")
#backpropogation()
#This method uses the backpropogation algorithm to determine the delta values of each node in order to later change weights(if the current test was wrong)
#Parameters:
# inputLayer -> The input layer of the network
# hiddenLayer -> The hidden layer of the network
# outputLayer -> The outputLayer of the network
# expectedOutput -> the expected output of the test that was wrong
# returns: The neural network with new weights
def backPropogation(inputLayer,hiddenLayer,outputLayer, expectedOutput):
    delta = 0
    nodeIndex = 0 #iterate to determine which error values go to which output node
    for outputNode in outputLayer: #determine delta values at output first, then work backwards
        errorValue =  int(expectedOutput[nodeIndex]) - outputNode.value #expected - actual
        outputNode.delta = FPrime(outputNode.value)*errorValue
        nodeIndex += 1
    
    #determine delta values in hidden layers
    biasNode = True
    for hiddenNode in hiddenLayer: #sigmoid bias values
        nodeIndex = 0
        delta = 0
        if biasNode == True: #avoid bias nodes, we don't want to change their values
            for outputNode in outputLayer: 
                delta += (FPrime(sigmoid(hiddenNode.value)) * outputNode.delta * hiddenNode.weights[nodeIndex]) 
                nodeIndex += 1
            biasNode = False
            hiddenNode.delta = delta
        else:
            for outputNode in outputLayer:
                delta += (FPrime(hiddenNode.value) * outputNode.delta * hiddenNode.weights[nodeIndex])
                nodeIndex += 1
            hiddenNode.delta = delta
    return(changeWeights(inputLayer,hiddenLayer,outputLayer))
    #don't need delta in input layers, can just go on to changing weights now

#changeWeights()
#After the delta values are determined, this method will use them to actually change the weights of the input and hidden layers
#Parameters:
# inputLayer -> The input layer of the network
# hiddenLayer -> The hidden layer of the network
# outputLayer -> The outputLayer of the network
# returns -> The whole network with changed weights
def changeWeights(inputLayer,hiddenLayer,outputLayer):
    alpha = 1 
    inputLayerCopy = copy.deepcopy(inputLayer)
    hiddenLayerCopy = copy.deepcopy(hiddenLayer)
    outputLayerCopy = copy.deepcopy(outputLayer)
    #change weights in input layer first
    for inputNode in inputLayerCopy:
        nodeIndex = 0
        for hiddenNode in hiddenLayerCopy:
            if(hiddenNode.value) == -1:
                continue
            else:
                inputNode.weights[nodeIndex] = inputNode.weights[nodeIndex] + (alpha * hiddenNode.delta * inputNode.value)
                nodeIndex += 1
    #change weights in hidden layer
    for hiddenNode in hiddenLayerCopy:
        nodeIndex = 0
        for outputNode in outputLayerCopy:
            hiddenNode.weights[nodeIndex] = hiddenNode.weights[nodeIndex] + (alpha * outputNode.delta * hiddenNode.value)
            nodeIndex += 1

    return(inputLayerCopy,hiddenLayerCopy, outputLayerCopy)

#passForward()
#This method will calculate the values of each node after given the values of the input nodes
#Parameters:
# inputLayer -> The input layer of the network
# hiddenLayer -> The hidden layer of the network
# outputLayer -> The outputLayer of the network
# returns -> an array [x,y,z] which will present the probability of it being a 1,8, or 9. x = probability of 1, y = probability of 7 etc.
def passForward(inputLayer, hiddenLayer, outputLayer): #reset values again somehow
    nodeNumber = 0
    for node in hiddenLayer:
        if nodeNumber !=0:
            node.value = 0 #initialize each value to zero to start the iteration
        nodeNumber+=1
    for node in outputLayer:
        node.value = 0 #same thing

    outputAnswers = []
    nodeNumber = 0
    #pass forward to hidden layer
    for hiddenNode in hiddenLayer: 
        if hiddenNode.value != -1:
            for inputNode in inputLayer: #need to not include bias node
                hiddenNode.value += (inputNode.value * inputNode.weights[nodeNumber]) 
            nodeNumber += 1
            hiddenNode.value = sigmoid(hiddenNode.value)
    #pass forward to output layer        
    nodeNumber = 0
    for outputNode in outputLayer:
        for hiddenNode in hiddenLayer:
            outputNode.value += (hiddenNode.value * hiddenNode.weights[nodeNumber])
        nodeNumber += 1
        outputNode.value = sigmoid(outputNode.value)
        outputAnswers.append(outputNode.value) #[0] = 1 [1] = 8 [2] = 9
    return(outputAnswers)
x = ["GRR", "train.txt", "train_output.txt"]
initializeNetwork(x)