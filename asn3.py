from operator import concat, index, neg
from os import system, write
import random
import numpy
import sys
import copy

# Output is 0.12 and 0.2, but the actual output is 1 and 0. We will calculate the errors, and the w values will be adjusted off them (big error, big weight change and if small error, small weigh change)
# Have to adjust weight values for hidden layers too, but can not calculate errors the same way in the hidden layers since they dont have a right value
# **She provided initial weight values** BIAS NODES ARE at the input AND hidden layers 
# 
# After calculating the amount of error (1-0.12) = 0.88 SEEN As (z-y) z is target y is actual output
# Apply fprime(incoming value to Y) -> (w46Y4 + w56Y5) * (z - y) = delta
# w46 = w46 + Learning rate (given by me) *y4*delta6      y4 is calculated earlier, delta six is the delta above as it was for output 6
# w56 + L.R * y5 * delta6
# TO go further back and calculate delta 4, you get f'(w14Y1 + w24Y2 + w34Y3) x** w46 x delta6 ** This is the back propagation because we cant do x -y
# do same for delta 5, but with its corresponding weights, but you will now consider w56 * delta 6
# TO DO delta 1, f'(node*weight + node*weight of all inputs) x w14 x delta4, Carry the deltas backwards 
# SINCE f1 is back prooagated by f4 and f5, you have to do both
# NOW IT IS f'(node*weight + node*weight of all inputs) x w14 + delta 4 + (node*weight + node*weight of all inputs)) xw15 + delta 5
# Now we are at the inputs, wx11 + wx11 + (learning rate * x1 * delta1)

#Derivates, we want the derivative to be zero, Error is a parabolic fuction, and we want to find the minimum on that graph, which is where derivative is zero
#if our derivative (or slope) is positive, we have to reduce weight, and increase it if the slope is negative
#Error function does not necessarily have to be a parabola (like in the example), it could have many spots where the slope is zero, which could find a local minima (want global)
# Finding the minimum here is dependent on the initial setting of weights** 

# We need to check the gradient (derivative slope), we will move a small amount in the opposite direction of the gradient, and then recalculate the gradient in a new spot
# calculate derivative of error/derivative of w
# we have a negative now instead which means we have to take in the opposite direciton
# on fucked slide, y is target output hw is my output -> either way we will end up not needing the derivative 
#in the sigmoid function, g'(x) = g(x) * 1-g(x)
#IF we use the sigmoid function as our logistic function, then we can calculate the weight using less values
class Node:
    def __init__(self, value) -> None:
        self.value = value
        self.weights = []
        self.delta = 0 #*** Can this be zero? ***
        self.sigmoid = None


def sigmoid(value):
    return 1/(1+ numpy.exp(neg(value)))

def FPrime(sigmoidValue):
    return(sigmoidValue*(1-sigmoidValue))

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


def initializeWeights(inputLayer, hiddenLayer, outputLayer, terminalArguement):
    for node in inputLayer:
        for i in range(10):
            node.weights.append(round(random.uniform(0,0.1),2)) #before it was 0,0.1
    for node in hiddenLayer:
        for i in range(3):
            node.weights.append(round(random.uniform(0,0.05),2))  #before it was 0,0.05
    trainNetwork(inputLayer, hiddenLayer, outputLayer, terminalArguement)
    
def writeToFile(expectedOutputIndex, ActualOutputIndex):
    file = open("D:/Programming/Repositories/BackpropogationDigitRecognition/BackpropogationDigitRecognition/train_output.txt","a")#desktop
    #file = open("D:/Programming/Repo/backPropogationDigitRecognition/BackpropogationDigitRecognition/train_output.txt", "a") #laptop
    if(ActualOutputIndex == 0):
        file.write("1")
    elif(ActualOutputIndex == 1):
        file.write("8")
    elif(ActualOutputIndex == 2):
        file.write("9")

    if(expectedOutputIndex == 0):
        file.write("                           1\n")
    elif(expectedOutputIndex == 1):
        file.write("                           8\n")
    elif(expectedOutputIndex == 2):
        file.write("                           9\n")
    file.close() #have to put the counter somehwere I guess?

def trainNetwork(inputLayer, hiddenLayer, outputLayer, terminalArguement):
    print(inputLayer[0].weights)
    fileToWrite = open("D:/Programming/Repositories/BackpropogationDigitRecognition/BackpropogationDigitRecognition/train_output.txt","w")#desktop
    #fileToWrite = open("D:/Programming/Repo/backPropogationDigitRecognition/BackpropogationDigitRecognition/train_output.txt", "w") #laptop
    fileToWrite.write("my_predicted_digit   target(correct digit)")
    fileToWrite.close()
    answer = []
    actualOutput = -1
    expectedOutput = -1
    correctlyClassified = 0
    numberOfClassifications = 0
    file = open("D:/Programming/Repositories/BackpropogationDigitRecognition/BackpropogationDigitRecognition/"+terminalArguement[0],"r")#desktop
    #file = open("D:/Programming/Repo/backPropogationDigitRecognition/BackpropogationDigitRecognition/"+terminalArguement[0],"r") #laptop

    lines = file.readlines()
    for line in lines:
        answer = []
        indexInput = 1
        numberOfClassifications += 1 #per input
        for word in line.split():
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
                    if expectedOutput == actualOutput.index(max(actualOutput)):#the program correctly guessed the output
                        correctlyClassified += 1
                 #       print("Correct")
                        continue
                    else:
                #        print("Incorrect")
                        newValues = backPropogation(inputLayer, hiddenLayer, outputLayer, answer)

    print(correctlyClassified / numberOfClassifications)
    if(correctlyClassified / numberOfClassifications < 0.9):
        trainNetwork(newValues[0], newValues[1], newValues[2], terminalArguement)
    else:
        testNetwork(newValues[0], newValues[1], newValues[2], terminalArguement[1])

def testNetwork(inputLayer, hiddenLayer, outputLayer, terminalArguement):
    file = open("D:/Programming/Repositories/BackpropogationDigitRecognition/BackpropogationDigitRecognition/"+terminalArguement,"r")#desktop
    #file = open("D:/Programming/Repo/backPropogationDigitRecognition/BackpropogationDigitRecognition/"+terminalArguement[0],"r") #laptop
    fileToWrite = open("D:/Programming/Repositories/BackpropogationDigitRecognition/BackpropogationDigitRecognition/test_output.txt","w")#desktop
    #file = open("D:/Programming/Repo/backPropogationDigitRecognition/BackpropogationDigitRecognition/train_output.txt", "a") #laptop
    lines = file.readlines()
    for line in lines:
        indexInput = 1
        for word in line.split():
            inputLayer[indexInput].value = int(float(word))
            indexInput += 1
        actualOutput = passForward(inputLayer, hiddenLayer, outputLayer)
        numberToWrite = actualOutput.index(max(actualOutput))
        if(numberToWrite == 0):
            fileToWrite.write("1\n")
        elif(numberToWrite == 1):
            fileToWrite.write("8\n")
        elif(numberToWrite == 2):
            fileToWrite.write("9\n")
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
        if biasNode == True:
            for outputNode in outputLayer: 
                delta += (FPrime(sigmoid(hiddenNode.value)) * outputNode.delta * hiddenNode.weights[nodeIndex]) ##probably wrong
                nodeIndex += 1
            biasNode = False
            hiddenNode.delta = delta
        else:
            for outputNode in outputLayer:
                delta += (FPrime(hiddenNode.value) * outputNode.delta * hiddenNode.weights[nodeIndex]) ##probably wrong
                
                nodeIndex += 1
            hiddenNode.delta = delta
    return(changeWeights(inputLayer,hiddenLayer,outputLayer))
    #don't need delta in input layers, can just go on to changing weights now


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

    for hiddenNode in hiddenLayerCopy:
        nodeIndex = 0
        for outputNode in outputLayerCopy:
            hiddenNode.weights[nodeIndex] = hiddenNode.weights[nodeIndex] + (alpha * outputNode.delta * hiddenNode.value)
            nodeIndex += 1

    return(inputLayerCopy,hiddenLayerCopy, outputLayerCopy)

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
    #for i in outputLayer:
    #   print(i.value)
    return(outputAnswers)
    
#print(round(random.random(),2))
initializeNetwork(["train.txt", "test.txt"])
#forwardPassingNetwork(sys.argv)

inputLayer = []
hiddenLayer = []
outputLayer = []
inputLayer.append(Node(-1))
inputLayer.append(Node(1))
inputLayer.append(Node(0))
inputLayer.append(Node(1))
for nodes in inputLayer:
    for x in range(3):
        nodes.weights.append(1)

hiddenLayer.append(Node(-1))
hiddenLayer.append(Node(None))
hiddenLayer.append(Node(None))
for node in hiddenLayer:
    for y in range(3):
        node.weights.append(1)
        
outputLayer.append(Node(None))
outputLayer.append(Node(None))
outputLayer.append(Node(None))




#print(passForward(inputLayer, hiddenLayer, outputLayer))

#passForward(inputLayer,hiddenLayer, outputLayer)

#backPropogation(inputLayer,hiddenLayer,outputLayer,[], ['0','0','1'])
