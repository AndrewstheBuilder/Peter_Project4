# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

import util
import classificationMethod
import math

from collections import OrderedDict

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1 # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
        self.conditionalProbTable = OrderedDict()
        self.priorProb = OrderedDict()

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(set([ f for datum in trainingData for f in list(datum.keys()) ]));

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def buildConditionalProbTableForEachK(self,kgrid,pixelCount):
        '''
        Builds a data structure for conditional probablity table for each K
        :return data structure
        '''
        self.features.sort() #sort features so it looks like this [(0,0),(0,1)...(1,0),(1,1),(1,2)...]
        conditionalProbforK = OrderedDict()
        for k in kgrid:
            conditionalProbforK[k] = OrderedDict()
            conditionalProbTable = conditionalProbforK[k]
            for label in self.legalLabels:
                conditionalProbTable[label] = OrderedDict()
                featureTable =  conditionalProbTable[label]
                featuresForLabel = pixelCount[label]
                for feature in self.features:
                    countFeatureIs0 = featuresForLabel[feature+(0,)]
                    countFeatureIs1 = featuresForLabel[feature+(1,)]
                    smoothedCondProbFor0 = (countFeatureIs0+k)/((countFeatureIs0+k)+(countFeatureIs1+k))
                    smoothedCondProbFor1 = (countFeatureIs1+k)/((countFeatureIs0+k)+(countFeatureIs1+k))
                    featureTable[feature] = [smoothedCondProbFor0,smoothedCondProbFor1] #contains probability that feature is a 0 or 1
        return conditionalProbforK

    def evaluateBestK(self, kgrid, conditionalProbForK, validationData, validateKeysDict):
        '''
        Evaluates which k's conditional probability table gives best accuracy with validationData
        Sets self.conditionalProbTable to best K conditional probability table
        :validateKeysDict - dictionary of indexes of labels occurences in validationData
        '''
        guesses = OrderedDict() #guesses returned for each k conditional probability table
        for k in kgrid:
            table = conditionalProbForK[k]
            guesses[k] = []
            self.conditionalProbTable = table
            for datum in validationData:
                posterior = self.calculateLogJointProbabilities(datum)
                guesses[k].append(posterior.argMax())

        #evaluate accuracy of guesses and assign best K conditional probability table to self.conditionalProbTable
        bestK = 0
        bestAccuracy = 0.00
        for k in kgrid:
            correct = 0
            incorrect = 0
            for label,indexList in validateKeysDict.items():
                guessList = guesses[k]
                for index in indexList:
                    if(guessList[index] == label):
                        correct += 1
                    else:
                        incorrect += 1
            currAccuracy = correct/(correct+incorrect)
            if(bestAccuracy < currAccuracy ):
                bestAccuracy = currAccuracy
                bestK = k
        self.conditionalProbTable = conditionalProbForK[bestK] #set conditional table to best conditional table


    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """

        "*** YOUR CODE HERE ***"
        trainingKeysDict = OrderedDict()
        validateKeysDict = OrderedDict()
        trainDataFeatureCount = OrderedDict()
        for label in self.legalLabels:
            trainingKeysDict[label] = [index for index in range(len(trainingLabels)) if trainingLabels[index] == label]
            validateKeysDict[label] = [index for index in range(len(validationLabels)) if validationLabels[index] == label]
        for label,indexList in trainingKeysDict.items():
            trainDataFeatureCount[label] = util.Counter()
            self.priorProb[label] = len(indexList)/len(trainingLabels)
            for index in indexList:
                for pixel in trainingData[index]:
                    if(trainingData[index][pixel] == 1):
                        trainDataFeatureCount[label][pixel+(1,)] += 1
                    elif(trainingData[index][pixel] == 0):
                        trainDataFeatureCount[label][pixel+(0,)] += 1

        #
        #choose the best k by creating conditional probability table for each k then running it against validation set
        #

        #build conditional probability table for each k
        conditionalProbforK = self.buildConditionalProbTableForEachK(kgrid,trainDataFeatureCount)

        #evaluate best K conditional probability table inside of conditionalProbforK
        self.evaluateBestK(kgrid, conditionalProbforK,validationData,validateKeysDict)

    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        logJoint = util.Counter()
        "*** YOUR CODE HERE ***"
        self.features.sort()
        for label in self.legalLabels:
            featureProbsForLabel = self.conditionalProbTable[label]
            condProbCounter = 0 #add all of the conditional probabilties
            for feature in self.features:
                res = datum[feature] #res should be 0 or 1
                condProbCounter += math.log(featureProbsForLabel[feature][res])
            logJoint[label] = math.log(self.priorProb[label]) + condProbCounter
        return logJoint

    def findHighOddsFeatures(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
                P(feature=1 | label1)/P(feature=1 | label2)

        Note: you may find 'self.features' a useful way to loop through all possible features
        """
        featuresOdds = []

        "*** YOUR CODE HERE ***"
        # print('label1',label1)
        # print('label2',label2)
        util.raiseNotDefined()
        return featuresOdds
