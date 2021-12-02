from operator import concat, index, neg
from os import system
import random
import numpy
import sys

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

def sigmoid(value):
    return 1/(1+ numpy.exp(neg(value)))

def initializeNetwork(terminalArguement):
    inputLayer = []
    hiddenLayer = []
    outputLayer = []
    inputLayer.append(Node(-1)) #bias node for input layer
    hiddenLayer.append(Node(-1)) #bias node for hidden layer
    for inputNode in range(256): #number of input nodes
        inputLayer.append(Node(-1))
    for hiddenNode in range(10): #number of hidden nodes
        hiddenLayer.append(Node(-1))
    for outputNode in range(3): #number of output nodes
        outputLayer.append(Node(-1)) 
    
    initializeWeights(inputLayer, hiddenLayer, outputLayer, terminalArguement)


def initializeWeights(inputLayer, hiddenLayer, outputLayer, terminalArguement):
    for node in inputLayer:
        for i in range(10):
            node.weights.append(round(random.random(),2)) 
    for node in hiddenLayer:
        for i in range(3):
            node.weights.append(round(random.random(),2))
    trainNetwork(inputLayer, hiddenLayer, outputLayer, terminalArguement)
    


def trainNetwork(inputLayer, hiddenLayer, outputLayer, terminalArguement):
    answer = []
    file = open("D:/Programming/Repo/backPropogationDigitRecognition/BackpropogationDigitRecognition/"+terminalArguement[0],"r")
    outputFile = open("D:/Programming/Repo/backPropogationDigitRecognition/BackpropogationDigitRecognition/train_output.txt","w")
    lines = file.readlines()
    indexInput = 1
    for line in lines:
        indexInput = 1
        for word in line.split():
            if(len(word)) == 6: #making sure we dont include the last three numbers in each line
                inputLayer[indexInput].value = int(float(word))
                indexInput += 1
            else: ##getting test result
                answer.append(word)
                if len(answer) == 3: #values have been inputted
                    testRound = passForward(inputLayer, hiddenLayer, outputLayer, terminalArguement)
   

def passForward(inputLayer, hiddenLayer, outPutLayer, terminalArguement):
    nodeNumber = 0
    for hiddenNode in hiddenLayer: 
        if hiddenNode.value != -1:
            for inputNode in inputLayer: #need to not include bias node
                hiddenNode.value += (inputNode.value * inputNode.weights[nodeNumber])
            nodeNumber += 1
            hiddenNode.value = sigmoid(hiddenNode.value)
    for x in hiddenLayer:
        print(x.value)

#print(round(random.random(),2))
initializeNetwork(["train.txt", "otherfile"])
#forwardPassingNetwork(sys.argv)
