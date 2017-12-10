from math import sqrt
import numpy as np


def cosine(dictA, dictB):
	'''
		Cosine similarity between two dictionaries

		Arguments:
			dictA 	- dictionary
			dictB 	- dictionary
	'''
	intersection = [o for o in dictA if o in dictB]
	if len(intersection) != 0:
		AB = sum([dictA[o] * dictB[o] for o in intersection])
		normA = sqrt(sum([dictA[o] ** 2 for o in dictA]))
		normB = sqrt(sum([dictB[o] ** 2 for o in dictB]))
		denominator = normA * normB
		if denominator == 0:
			return -1
		return AB / denominator
	else:
		return 0

def jaccard(dictA, dictB):
	'''
		Jaccard similarity between two dictionaries

		Arguments:
			dictA 	- dictionary
			dictB 	- dictionary
	'''
	intersectionCardinality = sum([1 for o in dictA if o in dictB])
	unionCardinality = len(dictA) + len(dictB) - intersectionCardinality
	if unionCardinality == 0:
		return -1
	else:
		return intersectionCardinality / unionCardinality

def pearson(dictA, dictB):
	'''
		Pearson similarity between two dictionaries

		Arguments:
			dictA 	- dictionary
			dictB 	- dictionary
	'''
	intersection = [o for o in dictA if o in dictB]
	if len(intersection) == 0:
		return 0
	meanOfA = np.mean([dictA[o] for o in dictA.keys()])
	meanOfB = np.mean([dictB[o] for o in dictB.keys()])
	numerator = sum([(dictA[o] - meanA) * (dictB[o] - meanB) for o in intersection])
	deviationOfA = sqrt(sum([(dictA[o] - meanOfA) ** 2 for o in intersection]))
	deviationOfB = sqrt(sum([(dictB[o] - meanOfB) ** 2 for o in intersection]))
	if deviationOfA == 0 or deviationOfB == 0:
		return 0
	return numerator / (deviationOfA * deviationOfB)