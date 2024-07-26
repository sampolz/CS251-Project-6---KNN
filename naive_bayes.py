'''naive_bayes_multinomial.py
Naive Bayes classifier with Multinomial likelihood for discrete features
Sam Polyakov
CS 251: Data Analysis and Visualization
Spring 2024
'''
import numpy as np

from classifier import Classifier

class NaiveBayes(Classifier):
    '''Naive Bayes classifier using Multinomial likeilihoods (discrete data belonging to any number of classes)'''
    def __init__(self, num_classes):
        '''Naive Bayes constructor

        TODO:
        - Call superclass constructor
        - Add placeholder instance variables the class prior probabilities and class likelihoods (assigned to None).
        You may store the priors and likelihoods themselves or the logs of them. Be sure to use variable names that make
        clear your choice of which version you are maintaining.
        '''
        self.priors = None
        self.likelihoods = None
        Classifier.__init__(self, num_classes)

    def get_priors(self):
        '''Returns the class priors (or log of class priors if storing that)'''
        return self.priors

    def get_likelihoods(self):
        '''Returns the class likelihoods (or log of class likelihoods if storing that)'''
        return self.likelihoods

    def train(self, data, y):
        '''Train the Naive Bayes classifier so that it records the "statistics" of the training set:
        class priors (i.e. how likely an email is in the training set to be spam or ham?) and the
        class likelihoods (the probability of a word appearing in each class — spam or ham)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        TODO:
        - Compute the class priors and class likelihoods (i.e. your instance variables) that are needed for
        Bayes Rule. See equations in notebook.
        '''
        num_samps, num_features = data.shape
        class_counts = np.bincount(y.astype(int))
        
        self.priors = class_counts / num_samps
        self.likelihoods = np.zeros((self.num_classes, num_features))
        
        for c in range(self.num_classes):
            class_data = data[y == c]
            Tcw = np.sum(class_data, axis=0)
            Tc = np.sum(Tcw)
            self.likelihoods[c] = (Tcw + 1) / (Tc + num_features)


    def predict(self, data):
        '''Combine the class likelihoods and priors to compute the posterior distribution. The
        predicted class for a test sample from `data` is the class that yields the highest posterior
        probability.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each test data sample.

        TODO:
        - For the test samples, we want to compute the log of the posterior by evaluating
        the the log of the right-hand side of Bayes Rule without the denominator (see notebook for
        equation). This can be done without loops.
        - Predict the class of each test sample according to the class that produces the largest
        log(posterior) probability (hint: this can also be done without loops).

        NOTE: Remember that you are computing the LOG of the posterior (see notebook for equation).
        NOTE: The argmax function could be useful here.
        '''
        num_test_samps, num_features = data.shape
    
        log_post = np.zeros((num_test_samps, self.num_classes))
        
        for c in range(self.num_classes):
            log_prior = np.log(self.priors[c])
            log_likelihoods = np.log(self.likelihoods[c])
            sum = np.sum(data * log_likelihoods, axis=1)
            log_post[:, c] = log_prior + sum
        
        predicted_class = np.argmax(log_post, axis=1)
        
        return predicted_class
