######################################
# > Ewwww Spam Filter
# > SpamFilter.py
# > Created by Uygur Kıran on 2021/07/29.
######################################

######################################
## dependencies ##
######################################
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import  train_test_split
from sklearn.naive_bayes import MultinomialNB

######################################
## model ##
######################################
class SpamFilter:
    ## PUBLIC METHODS ##
    def __init__(self, mailDataJSONFile = "./data/EmailTexts.json",
                 vectorizer = CountVectorizer(stop_words="english"),
                 msgColName = "MESSAGE", labelColName = "CATEGORY"):
        self.__IS_SPAM = 1

        ## prepare data ##
        self.__data = self.__getMailData(mailDataJSONFile)
        self.__vectorizer = vectorizer
        allFeatures = vectorizer.fit_transform(self.__data[msgColName])

        ## split ##
        self.__xTrain, self.__xTest, self.__yTrain, self.__yTest = \
            train_test_split(allFeatures, self.__data[labelColName],
                             test_size=0.2, random_state=88)

        ## create classifier ##
        self.__classifier = MultinomialNB()
        self.__classifier.fit(self.__xTrain, self.__yTrain)

    def isSpam(self, msg):
        """
        :param msg: Mail body as a string -or- Mail bodies as a list of strings
        :return: bool or [bool]
        """
        justStr = False
        if isinstance(msg, str):
            msg = [msg]
            justStr = True
        elif not isinstance(msg, list):
            return None

        termMatrix = self.__vectorizer.transform(msg)
        res = self.__classifier.predict(termMatrix)

        if justStr:
            return bool(res[0] == self.__IS_SPAM)
        else:
            return [(i > 0) for i in res]

    def evaluateModel(self):
        from sklearn.metrics import f1_score
        ## check accuracy ##
        print("*", "MODEL EVALUATION", "*" * 20)
        correctResCount = (self.__yTest == self.__classifier.predict(self.__xTest)).sum()
        incorrectResCount = self.__yTest.size - correctResCount
        print(f"\t- {correctResCount} test emails classified correctly.")
        print(f"\t- {incorrectResCount} test emails classified incorrectly.")
        ## return f-score ##
        f1Score = f1_score(self.__yTest, self.__classifier.predict(self.__xTest))
        print("\t- F-Score is {:.4f}.".format(f1Score))
        print("*" * 39)
        return f1Score

    ## PRIVATE METHODS ##
    def __getMailData(self,path):
        data = pd.read_json(path)
        data.sort_index(inplace=True)
        return data
