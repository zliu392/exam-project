# import libraries
from scipy.optimize import linprog
from math import sqrt
from collections import defaultdict
import numpy as np
import random
import copy
import math
import matplotlib.pyplot as plt
import timeit
import pandas as pd
from decimal import Decimal
from scipy import stats
import time
random.seed(42)########################################

# initialize constants

# hardcoded constants for # of subjects, treatments, capacity, budget
num_subjects = 1540 # i
num_treatments = 2 # t
capacity_matrix = [663, 877]
budget = 100
epsilon = 0.2 # has to be less than 0.5
rct_treatment_probabilities = [capacity_matrix[0]*1.0/num_subjects, capacity_matrix[1]*1.0/num_subjects]
budget_matrix = [budget] * num_subjects

# Scaling factor for alpha, beta to set new prices -- hyperparameters
alpha_scaling_factor = 0.5
beta_scaling_factor = budget/50

# Initialize alpha, beta assumed to be positive
def init_alpha():
    alpha = np.asarray([random.randint(-budget, 0) for i in range(num_treatments)])
    return alpha
def init_beta():
    beta = np.asarray([random.randint(-budget, budget) for i in range(num_treatments)])
    return beta

# Price vector pi(i,t) = alpha(t) * pte(i,t) + beta(t). Dimensions num_subjects * num_treatments
def get_price_matrix(alpha, beta):
    price_matrix = [[(alpha[index] * pte_t + beta[index]) for index, pte_t in enumerate(pte)] for pte in pte_matrix]
    price_matrix = np.asarray(price_matrix)
    return price_matrix

# Demand p*(i,t) matrix. Solve LP to get values. Dimensions num_subjects * num_treatments
def get_demand_matrix(price_matrix):
    x0_bounds = (0,1)
    x1_bounds = (0,1)

    # dummy first row
    demand_matrix = np.ndarray((num_subjects,num_treatments), float)
    for i in range(num_subjects):
        # Constraints:
        # 1. <p*(i), pi(i)> <= b(i) for every subject i
        # 2. sum of all p*(t) = prob_threshold for every subject i
        coefficients = price_matrix[i]
        thresholds = budget_matrix[i]

        result = linprog(c=-wtp_matrix[i],
                         A_ub = [[coefficients[0], coefficients[1]]],
                         b_ub = thresholds,
                         A_eq = [[1,1]],
                         b_eq = 1,
                         bounds = (x0_bounds, x1_bounds))
        demand_matrix[i] = result.x

    # change the type to matrix
    demand_matrix = np.asmatrix(demand_matrix)
    # compute the bounded probability (as in slide page 7)
    min_prob0 = float(min(demand_matrix[:,0]))
    max_prob0 = float(max(demand_matrix[:,0]))
    q1 = (epsilon - min_prob0) / (rct_treatment_probabilities[0] - min_prob0)
    q2 = (1 - epsilon - max_prob0) / (rct_treatment_probabilities[0] - max_prob0)
    min_prob1 = float(min(demand_matrix[:,1]))
    max_prob1 = float(max(demand_matrix[:,1]))
    q3 = (epsilon - min_prob1) / (rct_treatment_probabilities[1] - min_prob1)
    q4 = (1 - epsilon - max_prob1) / (rct_treatment_probabilities[1] - max_prob1)
    q = max(q1,q2,q3,q4)
    demand_matrix[:,0] = (1-q) * demand_matrix[:,0] + q * rct_treatment_probabilities[0]
    demand_matrix[:,1] = (1-q) * demand_matrix[:,1] + q * rct_treatment_probabilities[1]
    demand_matrix = np.asarray(demand_matrix)

    return demand_matrix

# Treatment_demand(t) = sum of demand(t) across all i. Dimensions 1 * num_treatments
def get_treatment_demand_matrix(demand_matrix):
    treatment_demand_matrix = np.zeros(num_treatments)
    for subject in range(num_subjects):
        for treatment in range(num_treatments):
            treatment_demand_matrix[treatment] += demand_matrix[subject, treatment]
    return treatment_demand_matrix

# Excess_demand(t) = treatment_demand(t) - capacity(t). Dimensions 1 * num_treatments
def get_excess_demand_matrix(treatment_demand_matrix):
    excess_demand_matrix = treatment_demand_matrix - capacity_matrix
    return excess_demand_matrix

# Clearing error in market = sqrt(sum of excess_demand(t)^2 for every treatment t)
def get_clearing_error(excess_demand_matrix):
    # If demand is satisfied everywhere and total capacity > number of subjects, no clearing error
    if all(excess <= 0 for excess in excess_demand_matrix):
        print "get_clearing_error: Market clear, no clearing error!"
        return 0
    else:
        clearing_error = sqrt(sum([excess**2 for excess in excess_demand_matrix]))
        clearing_error = clearing_error / sum(capacity_matrix)
        print "get_clearing_error: Clearing error:", clearing_error
        return clearing_error

# Recalibrate alpha, beta values to set new prices
def get_alpha_new(alpha, excess_demand_matrix):
    alpha_new = alpha
    return alpha_new

def get_beta_new(beta, excess_demand_matrix):
    beta_new = beta + excess_demand_matrix * beta_scaling_factor
    return beta_new

# Find market clearing price vector. The objective is to change alpha and beta values so that we reduce clearing error
def clear_market():

    # Initialize market prices and demand
    alpha = init_alpha()
    beta = init_beta()
    price_matrix = get_price_matrix(alpha, beta)
    demand_matrix = get_demand_matrix(price_matrix)
    excess_demand_matrix = get_excess_demand_matrix(get_treatment_demand_matrix(demand_matrix))
    clearing_error = get_clearing_error(excess_demand_matrix)

    # clearing error is percentage of total capacity so we want the market to clear at 1%
    clearing_error_threshold = 0.01
    threshold_iterations = 10
    iterations = 0
    minimum_clearing_error = clearing_error
    alpha_star = 0
    beta_star = 0

    # Set new prices to clear market
    while True:
        if iterations > threshold_iterations:
            # new search start
            alpha = init_alpha()
            beta = init_beta()
            iterations = 0
            print "new search start"
        else:
            # continue down current search
            alpha = get_alpha_new(alpha, excess_demand_matrix)
            beta = get_beta_new(beta, excess_demand_matrix)

        price_matrix = get_price_matrix(alpha, beta)
        demand_matrix = get_demand_matrix(price_matrix)
        excess_demand_matrix = get_excess_demand_matrix(get_treatment_demand_matrix(demand_matrix))
        clearing_error = get_clearing_error(excess_demand_matrix)

        # Store parameter values for minimum clearing error
        if clearing_error < minimum_clearing_error:
            minimum_clearing_error = clearing_error
            alpha_star = alpha.copy()
            beta_star = beta.copy()
        # cleared the market!
        if minimum_clearing_error < clearing_error_threshold:
            break
        iterations += 1

    print "Minimum clearing error:", minimum_clearing_error
    print "Alpha_star:", alpha_star
    print "Beta star:", beta_star
    return (minimum_clearing_error, alpha_star, beta_star)

def simulate():
    while True:
        min_error, alpha_star, beta_star = clear_market()
        price_star = get_price_matrix(alpha_star, beta_star)
        demand_star = get_demand_matrix(price_star)

        control_probs = [demand_star_i[0] for demand_star_i in demand_star]
        treatment_probs = [demand_star_i[1] for demand_star_i in demand_star]

        if (min_error < 0.01):
            print "cleared market!"
            break
    return demand_star

# dict of form {dataset : demand_star}
# demand_star is a list of [control_demand, treatment_demand]
# every dataset is mapped to the market clearing probability distribution
demand_dict = {}
# dict of form {dataset : # unique groups}
num_input_groups_dict = {}
num_output_groups_dict = {}
problem_datasets = []

#  Sanity check error counter
sanity_check = 0
bound_violation = -100

start_dataset, end_dataset = 1,1001
for d in range(start_dataset, end_dataset):

    # load PTE data
    pte_df = pd.read_csv("input/WTP_HTE_forPythonEXaMalgorithm/PTE_dia_all_"+str(d)+"_COARSE.csv")


    # load WTP data
    wtp_df = pd.read_csv("input/WTP_HTE_forPythonEXaMalgorithm/WTP_wdays_all_"+str(d)+"_COARSE.csv")
#    wtp_df = pd.read_csv("temp/08_WTP10_"+str(d)+"_coarse.csv")
#    wtp_df = pd.read_csv("temp/08_WTP100_"+str(d)+"_coarse.csv")
#    wtp_df = pd.read_csv("temp/08_WTP1000_"+str(d)+"_coarse.csv")
#    wtp_df = pd.read_csv("temp/08_WTP100uplus_"+str(d)+"_coarse.csv")
#    wtp_df = pd.read_csv("temp/08_WTP100uminus_"+str(d)+"_coarse.csv")

    pte_matrix = [[0, i] for i in pte_df['PTE'].values.tolist()]
    wtp_matrix = [[0, i] for i in wtp_df['WTP'].values.tolist()]

    # Convert lists to np.array type for computation
    wtp_matrix = np.array(wtp_matrix)
    pte_matrix = np.array(pte_matrix)
    budget_matrix = np.array(budget_matrix)
    capacity_matrix = np.array(capacity_matrix)

    # solve market, add to dict
    demand_star = simulate()
    demand_dict[d] = demand_star.tolist()

    control_probs = [demand_star_i[0] for demand_star_i in demand_star]
    treatment_probs = [demand_star_i[1] for demand_star_i in demand_star]

    # sanity check
    # make dictionary to idenitfy subjects with same pte, wtp
    # {(pte, wtp) : [subject numbers]} -- get groups
    # now make sure that in each group, everyone has the same treatment and control assignment probability
    sanity_dict = defaultdict(list)
    for subject_num in range(len(wtp_matrix)):
        sanity_dict[(wtp_matrix[subject_num][1], pte_matrix[subject_num][1])].append(subject_num)
    num_input_groups_dict[d] = len(sanity_dict)

    for group in sanity_dict.values():
        if not all([treatment_probs[group[0]] == treatment_probs[subject_num] for subject_num in group]):
            print "problem"
        if not all([control_probs[group[0]] == control_probs[subject_num] for subject_num in group]):
            print "problem"

    # bounds sanity check
    b1 = min(control_probs)
    b2 = max(control_probs)
    b3 = min(treatment_probs)
    b4 = max(treatment_probs)
    if (b1< epsilon) or (b2>1-epsilon) or (b3<epsilon) or (b4>1-epsilon):
            problem_datasets.append(d)
            print "bounds not correct for", d
            print "min treatment probability is", Decimal(b3)
            print "max treatment probability is", Decimal(b4)
            sanity_check += 1
            bound_violation = max(Decimal(bound_violation), Decimal(epsilon - b1),
                                  Decimal(b2-1+epsilon), Decimal(epsilon - b3), Decimal(b4-1+epsilon))
            print "the largest bound violation till now is", Decimal(bound_violation)

    # count the number of unique values of p_it -- groups of subjects with same demand
    output_groups_dict = defaultdict(list)
    for i, demand_i in enumerate(demand_star):
        output_groups_dict[(demand_i[0], demand_i[1])].append(i)
    num_output_groups_dict[d] = len(output_groups_dict)

    print "finished dataset", d



print "bounds sanity check fails for", sanity_check, "dataset(s)"
print "max bound violation", Decimal(bound_violation)
# save the outputs
df_results = pd.DataFrame.from_dict(demand_dict)
df_results.info()
df_results.to_csv('input/df_results_coarse.csv')
#df_results.to_csv('input/df_results_mani10_coarse.csv')
#df_results.to_csv('input/df_results_mani100_coarse.csv')
#df_results.to_csv('input/df_results_mani1000_coarse.csv')
#df_results.to_csv('input/df_results_mani100uplus_coarse.csv')
#df_results.to_csv('input/df_results_mani100uminus_coarse.csv')
