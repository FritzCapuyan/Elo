import numpy as np
from scipy.optimize import minimize

def logistic(x, scale=400):
    return 1/(1+10**(x/scale))


def get_elo(competitor, elo_dict):
    try:
        competitor_score = elo_dict[competitor]
    except KeyError:
        elo_dict[competitor] = {'score':1000, 'num': 0, 'extra': 0}
        competitor_score = elo_dict[competitor]
    return competitor_score, elo_dict


def check_data(ids, results, extras=None):
    if extras is None:
        extras = [0 for x in results]
    if not len(ids) == len(results) == len(extras):
        raise BaseException("Elements not the same length")
    return zip(ids, results, extras)

def fit_elos(elo_df, K_num, K, extra_factor, return_type = "dict"):
    elo_dict = {}
    loglike = 0
    diff_list = []

    if return_type not in ['dict', 'list', 'log_loss']:
        raise BaseException("Set return_type as one of 'dict', 'list', 'log_loss'")

    for fight in elo_df:
        left_score, elo_dict = get_elo(fight[0][0], elo_dict)
        right_score, elo_dict = get_elo(fight[0][1], elo_dict)

        result = fight[1]
        
        expected = logistic((right_score['score']-extra_factor*right_score['extra'])-(left_score['score']-extra_factor*left_score['extra']))

        if return_type == 'list':
            diff_list.append((left_score['score']-extra_factor*left_score['extra'], right_score['score']-extra_factor*right_score['extra']))

        elo_dict[fight[0][0]]['num'] += 1
        elo_dict[fight[0][1]]['num'] += 1

        left_k = K[next((x[0] for x in enumerate(K_num) if x[1] > left_score['num']))]
        right_k = K[next((x[0] for x in enumerate(K_num) if x[1] > right_score['num']))]

        elo_dict[fight[0][0]]['score'] += left_k*(result-expected)
        elo_dict[fight[0][1]]['score'] -= right_k*(result-expected)

        if result == 1:
            loglike += np.log(expected)
            elo_dict[fight[0][0]]['extra'] = 0
            if fight[2] == 1:
                elo_dict[fight[0][1]]['extra'] += 1
        elif result == 0:
            loglike += np.log(1-expected)
            elo_dict[fight[0][1]]['extra'] = 0
            if fight[2] == 1:
                elo_dict[fight[0][0]]['extra'] += 1
    
    if return_type=='list':
        return diff_list
    elif return_type=='dict':
        return elo_dict
    else:
        return -loglike

def elo(ids, results, extras = None, return_type='dict', params = None):
    """
    ids: list of tuples of length 2 which represent unique ids of participants in the pairwise comparison

    results: list of values between 0 and 1 representing the score of the first id in the associated id pair. For example,
    the lists ids = [(1,2)] and results = [0] would mean that player 1 lost the first competition

    extras: optional list that is a placeholder for adding extra parameters, such as home field advantage. 

    return_type: one of 'dict', 'list', or 'log_loss'. Determines which results are returned. 'Dict' returns the dictionary with participant ids as keys
    and scores as values

    params: optional dictionary containing parameters to be used to fit elo score. Requires keys: 'K_num', 'K', 'extra'. The values supplied to 'K_num' 
    and 'K' must be the same length
    """
    elo_df = list(check_data(ids, results, extras))
    if params is None:
        res= minimize(lambda x: fit_elos(elo_df=elo_df, return_type='log_loss', K_num = [0, 5, 10, 15, 20], K=x[:-1], extra_factor=x[-1]), x0=[170, 170, 170, 170, 170, 50])
        return fit_elos(elo_df=elo_df, return_type=return_type, K_num = [0, 5, 10, 15, 20], K=res.x[:-1], extra_factor=res.x[-1])
    else:
        assert(len(params['K_num'])==len(params['K']))
        return fit_elos(elo_df=elo_df, return_type=return_type, K_num = params['K_num'], K=params['K'], extra_factor=params['extra'])