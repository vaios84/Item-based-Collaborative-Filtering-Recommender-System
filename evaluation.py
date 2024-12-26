from sklearn.metrics import ndcg_score, dcg_score
import numpy as np


def calculate_recall(K, predictions, ground_truth):
    predictions_at_K = predictions[:K]
    relevant_items = 500    # we consider as relevant items the top X of the gt list
    result = [i for i in predictions_at_K if i in ground_truth[:relevant_items]]
    enumerator = len(result)
    recall = enumerator / relevant_items
    return recall


def calculate_ndcg(true_ratings):
    # NDCG = Normalized Discounted Cumulative Gain
    # Relevance scores in output order in the predictions
    relevance_score = np.asarray([true_ratings])
    print(relevance_score)

    # Sort the ratings in descending order
    true_ratings.sort(reverse=True)
    # Relevance scores in Ideal order
    true_relevance = np.asarray([true_ratings])
    print(true_relevance)

    # DCG score
    dcg = dcg_score(true_relevance, relevance_score)
    # IDCG score
    idcg = dcg_score(true_relevance, true_relevance)

    # Normalized DCG score
    ndcg = dcg / idcg
    # print("nDCG score : ", ndcg)
    # or we can use the scikit-learn ndcg_score package
    # print("nDCG score (from function) : ", ndcg_score(true_relevance, relevance_score))
    return round(ndcg, 4)

