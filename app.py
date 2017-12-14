import pandas as pd
from datetime import datetime, timedelta
import time
from operator import itemgetter
import click

import similarity as sim
import recommender as r

_activity_data_train = None
_dealDetails_data_train = None
_detailItems_data_train = None
_activity_data_test = None
_dealDetails_data_test = None
_detailItems_data_test = None

def getUserItemDict(data):
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

def bestsellers(d = 30):
	'''
	 Makes list of besseller for last d (default 60) days

	 Arguments:
	 	d 	- number of days
	 '''
	lastDate = _activity_data_train['create_time'].max()
	dDaysBefore = lastDate - int(timedelta(days = d).total_seconds())
	_bestsellers = []
	_dict = getUserItemDict(_activity_data_train[_activity_data_train['create_time'] > dDaysBefore])
	t_dict = transposeDict(_dict)

	for item in t_dict:
		_bestsellers.append((item, sum([t_dict[item][user] for user in t_dict[item]])))
	return sorted(_bestsellers, key = itemgetter(1), reverse = True)[0:10]

def getItemsEndDateDict():
	'''
		Makes dictionary of coupon end times
	'''
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
	precision_sum = 0
	recall_sum = 0
	f1_sum = 0
	hitUsersCount = 0
	i = 0
	count = len(testDataSet)
	print('Creating recommendations for users...')
	for user in testDataSet:
		print('\r', i, '/', count, end = '')
		recommendations = [item[0] for item in recommender.recommendation(user, dates.get(user), n)]
		hit = sum([1 for item in testDataSet[user] if item in recommendations])
		precision = hit / n
		recall = hit / len(testDataSet[user])
		if hit > 0:
			hitUsersCount += 1
			f1 = 2 * precision * recall / (precision + recall)
		else:
			f1 = 0
		precision_sum += precision
		recall_sum += recall
		f1_sum += f1
		i += 1
	print('\n', 'Recommendation completed')
	print()
	print('--------------- EVALUATION ---------------')
	print('Precision: ', precision_sum / len(testDataSet))
	print('Recall: ', recall_sum / len(testDataSet))
	print('F1: ', f1_sum / len(testDataSet))
	print()
	print('Count of users: ', len(testDataSet))
	print('Hited users: ', hitUsersCount, '[', (hitUsersCount / count) * 100, '%]')
	print('------------------------------------------')

@click.command()
@click.option('--sim', prompt = 'similarity method', default = 'cosine')
def setSimilarityMethod(sim):
	if sim == 'cosine':
		return sim.cosine
	elif sim == 'pearson':
		return sim.pearson
	elif sim == 'jaccard':
		return sim.jaccard
	else:
		return None

if __name__ == '__main__':
	'''
		Here are loaded data and called all functions for making recommendations
	'''
	start = time.time()

	simmilarityMethod = setSimilarityMethod()

	# loading data from csv files using pandas module
	_activity_data_train = pd.read_csv('data/train_activity_v2.csv', sep = ',')
	_dealDetails_data_train = pd.read_csv('data/train_deal_details.csv', sep = ',')
	_detailItems_data_train = pd.read_csv('data/train_dealitems.csv', sep = ',')
	_activity_data_test = pd.read_csv('data/test_activity_v2.csv', sep = ',')
	_dealDetails_data_test = pd.read_csv('data/test_deal_details.csv', sep = ',')
	_detailItems_data_test = pd.read_csv('data/test_dealitems.csv', sep = ',')

	# makes data dictioranies
	userItem_data = getUserItemDict(_activity_data_train)
	itemUser_data = transposeDict(userItem_data)
	userItem_data_test = getUserItemDict(_activity_data_test)

	activityDates = getActivitiesCreateTime()

	ibcf = r.ItemBasedCF(userItem_data, itemUser_data, getItemsEndDateDict(), bestsellers())
	ibcf.buildItemSimilarityDict(n = 50, simmilarityMethod = sim.jaccard)

	evaluation(ibcf, userItem_data_test, activityDates, 10)
	print('===')
	print('Execution time: ', int(time.time() - start), ' seconds')