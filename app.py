import pandas as pd
import datetime
import time

import similarity as sim
import recommender as r

_activity_data_train = None
_dealDetails_data_train = None
_detailItems_data_train = None
_activity_data_test = None
_dealDetails_data_test = None
_detailItems_data_test = None

def getUserItemDic(data):
	'''
		Makes dictionary, which contains all items bought by user_id
		format of dictionary: {user_id: {dealitem_id: quantity, ...}, ...}

		Arguments:
			data 	- pandas dataFrame object
	'''
	_dict = {}
	for row in data.itertuples():
		_dict.setdefault(row.user_id, {})
		rating = 0
		if row.quantity > 5:
			rating = 5
		else:
			rating = row.quantity
		_dict[row.user_id][row.dealitem_id] = rating

	return _dict

def transposeDict(_dict):
	'''
		Makes a transposed dictionary (switch indicies)
		example: from User-Item dict makes a Item-User dict

		Arguments:
			_dict 	- dictionary
	'''
	t_dict = {}
	for o in _dict:
		for s in _dict[o]:
			t_dict.setdefault(s, {})
			t_dict[s][o] = _dict[o][s]

	return t_dict

def getItemsEndDateDict():
	_dict = {}
	for item in _detailItems_data_train.itertuples():
		_dict[item.id] = item.coupon_end_time
	return _dict

def getActivitiesCreateTime():
	'''
		Makes pandas series, which contains date of first user's activity in test data set
	'''
	return _activity_data_test.groupby('user_id')['create_time'].min()


def evaluation(recommender, testDataSet, dates, n = 10):
	'''
		Evaluation of recommender

		Arguments:
			recommender 	- recommender object
			testDataSet 	- dictionary
			n 				- count of recommendation for user
	'''
	itemPrecision_sum = 0
	hitUsersCount = 0
	i = 0
	count = len(testDataSet)
	for user in testDataSet:
		print(i, '/', count)
		recommendations = [item[0] for item in recommender.recommendation(user, dates.get(user), n)]
		hit = sum([1 for item in testDataSet[user] if item in recommendations])
		if hit > 0:
			hitUsersCount += 1
		itemPrecision = hit / n
		itemPrecision_sum += itemPrecision
		i += 1
	print('Precision: ', itemPrecision_sum / (len(testDataSet) * 10))
	print('Count of users: ', len(testDataSet))
	print('Hited users: ', hitUsersCount, '[', (hitUsersCount / count) * 100, '%]')

if __name__ == '__main__':
	'''
		Here are loaded data and called all functions for making recommendations
	'''
	start = time.time()

	# loading data from csv files using pandas module
	_activity_data_train = pd.read_csv('data/train_activity_v2.csv', sep = ',')
	_dealDetails_data_train = pd.read_csv('data/train_deal_details.csv', sep = ',')
	_detailItems_data_train = pd.read_csv('data/train_dealitems.csv', sep = ',')
	_activity_data_test = pd.read_csv('data/test_activity_v2.csv', sep = ',')
	_dealDetails_data_test = pd.read_csv('data/test_deal_details.csv', sep = ',')
	_detailItems_data_test = pd.read_csv('data/test_dealitems.csv', sep = ',')

	# makes data dictioranies
	userItem_data = getUserItemDic(_activity_data_train)
	itemUser_data = transposeDict(userItem_data)
	userItem_data_test = getUserItemDic(_activity_data_test)

	activityDates = getActivitiesCreateTime()

	ibcf = r.ItemBasedCF(userItem_data, itemUser_data, getItemsEndDateDict())
	ibcf.buildItemSimilarityDict(n = 50, simmilarityMethod = sim.jaccard)
	print(len(userItem_data.keys()))
	print(len(itemUser_data.keys()))

	evaluation(ibcf, userItem_data_test, activityDates, 10)

	print('time: ', time.time() - start)