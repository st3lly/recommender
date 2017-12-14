import abc
import numpy as np
from operator import itemgetter
import _pickle as pickle
import os

import similarity as sim

class AbstractRecommender(metaclass = abc.ABCMeta):
	'''
		Abstract class for recommenders, whos will inherit basic features from this class
	'''
	@abc.abstractmethod
	def buildItemSimilarityDict(self, simmilarityMethod = sim.cosine, n = 20):
		raise NotImplementedError

	@abc.abstractmethod
	def recommendation(self, userID, n = 10):
		raise NotImplementedError

class UserBasedCF(AbstractRecommender):
	'''
		There was not enough time for doing this class
	'''
	def buildItemSimilarityDict(self, simmilarityMethod = sim.cosine, n = 20):
		pass

	def recommendation(self, userID, n = 10):
		pass

class ItemBasedCF(AbstractRecommender):
	'''
		Simple Item-Based Collaborative Filtering class
	'''
	def __init__(self, userItem_data, itemUser_data, itemsEndDateDict, bestsellers):
		'''
			Initialize object
			
			Arguments:
				userItem_data 		- dictionary {userID : {itemID : quantity, ...}, ...}
				itemUser_data 		- dictionary {ttemID : {userID : quantity, ...}, ...}
				itemsEndDateDict 	- dictionary {item: endDate}
				bestsellers 		- list of top 10 bestsellers for d days (see app.bestellers function)

		'''
		self.__itemUser_data = itemUser_data
		self.__userItem_data = userItem_data
		self.__itemSimilarityDict = None
		self.__itemSimilarityDict_sum = None
		self.__bestsellers = bestsellers
		self.__itemsEndDateDict = itemsEndDateDict

	def similarItems(self, item, n, simmilarityMethod):
		'''
			Findes all similar items of item in argument and returns top n similar items, which are sorted descending

			Arguments:
				item 	- terget item ID
				n 		- count of similar items
		'''
		similarities = [(otherItem, simmilarityMethod(self.__itemUser_data[item], self.__itemUser_data[otherItem])) for otherItem in self.__itemUser_data if item != otherItem]
		return sorted(similarities, key = itemgetter(1), reverse = True)[0:n]

	def buildItemSimilarityDict(self, simmilarityMethod = sim.cosine, n = 20):
		'''
			Builds disctionary of similar items, which contains top-N similar items for each item
			format of dictionary: {item: {similarItem: simmilarity, ...}, ...}

			Arguments:
				simmilarityMethod 	- function, which will be used for calculate similarity (default cosine)
				n 					- count of similar items for item (defalut 20)
		'''
		fileName_is_dict = 'pickles/isd_' + simmilarityMethod.__name__ + '.pickle'
		fileName_sum_dict = 'pickles/isd_' + simmilarityMethod.__name__ + '_sum.pickle'
		print('Similarity method: ', simmilarityMethod.__name__)
		if not os.path.exists(fileName_is_dict):
			print('Item Similarity Dctionary file doesn\'t exist.')
			print('Building dictionary ...')
			is_dict = {}
			sum_dict = {}
			for item in self.__itemUser_data:
				is_dict.setdefault(item, {})
				sum_dict.setdefault(item, 0)
				correlations = self.similarItems(item, n, simmilarityMethod)
				for similarItem, similarity in correlations:
					is_dict[item][similarItem] = similarity
					sum_dict[item] += similarity
			self.__itemSimilarityDict = is_dict
			self.__itemSimilarityDict_sum = sum_dict
			print('Build completed')

			# Serialize dictionary for late use
			print('Serializing dictionaries for late use ...')
			with open(fileName_is_dict, 'wb') as handle:
				pickle.dump(self.__itemSimilarityDict, handle)

			with open(fileName_sum_dict, 'wb') as handle:
				pickle.dump(self.__itemSimilarityDict_sum, handle)
			
			print('Serialize completed')
		else:
			# Load serialized dictionary from file
			print('Loading Item Similarity Dctionary from file ...')
			with open(fileName_is_dict, 'rb') as handle:
				self.__itemSimilarityDict = pickle.load(handle)

			with open(fileName_sum_dict, 'rb') as handle:
				self.__itemSimilarityDict_sum = pickle.load(handle)
			
			print('Load complete')

	def checkItemDate(self, itemID, date):
		'''
			Checks item end date

			Arguments:
				item 	- ID of item
		'''
		if itemID in self.__itemsEndDateDict and self.__itemsEndDateDict[itemID] > date:
			return True
		else:
			return False

	def recommendation(self, userID, dealDate, n = 10):
		'''
			Makes topN recommendations for one user

			Arguments:
				user 	- ID of specific user
				n 		- count of  recommendations for user (default 10)
		'''
		if userID in self.__userItem_data:	# Known user, which had some purchases in the past
			predictions = []
			for candidate in self.__itemUser_data.keys():
				# If was the candidate item already bought by user, we can skip
				if candidate in self.__userItem_data[userID]:
					continue

				# If was the canditate expired, we can skip
				if not self.checkItemDate(candidate, dealDate):
					continue

				# If the candidate item was bought less than 10 times, we can skip
				if len(self.__itemUser_data[candidate]) < 10:
					continue

				correlations = self.__itemSimilarityDict[candidate]
				
				# Calculate predictions
				numerator = 0
				for item in self.__userItem_data[userID]:
					if item in correlations:
						# now we can calculate sccore for candidate item
						numerator += correlations[item] * self.__userItem_data[userID][item]	# Item similarity * item rating (purchases from user)

				if self.__itemSimilarityDict_sum[item] > 0:
					predictions.append((candidate, numerator / self.__itemSimilarityDict_sum[item]))
					#predictions.append((candidate, numerator))
				else:
					predictions.append((candidate, 0))

			predictions.sort(key = itemgetter(1), reverse = True)
			index = 0
			modifiedPredictions = []
			for item in predictions[0:n]:
				if item[1] == 0:
					modifiedPredictions.append(self.__bestsellers[index])
					index += 1
				else:
					modifiedPredictions.append(item)
			if index > 0:
				return modifiedPredictions
			else:
				return predictions[0:n]
		else:	# new user without previous purchases
			return self.__bestsellers
