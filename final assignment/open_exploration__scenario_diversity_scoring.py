import random
import pandas as pd
from sklearn import preprocessing
from concurrent.futures import ProcessPoolExecutor
import os
import functools
from scipy.spatial.distance import pdist, squareform
from scenario_diversity import find_maxdiverse_scenarios
import itertools
import numpy as np
from ema_workbench import Scenario

combined_df = pd.read_csv('output/base_case_results__100000_scenarios__combined_df.csv')
combined_df = combined_df.rename({'Unnamed: 0' : 'Run ID'}, axis=1)
print(combined_df.head())
outcomes_of_interest = ['A.4 Expected Annual Damage', 'A.4 Expected Number of Deaths',
                        'Total Expected Annual Damage', 'Total Expected Number of Deaths']
uncertainties = ['A.0_ID flood wave shape', 'A.1_Bmax', 'A.1_Brate', 'A.1_pfail', 'A.2_Bmax',
                 'A.2_Brate', 'A.2_pfail', 'A.3_Bmax', 'A.3_Brate', 'A.3_pfail', 'A.4_Bmax',
                 'A.4_Brate', 'A.4_pfail', 'A.5_Bmax', 'A.5_Brate', 'A.5_pfail',
                 'discount rate 0', 'discount rate 1', 'discount rate 2',]

indices = []
for idx, row in combined_df.iterrows():
    indices.append(idx) #row['Run ID'])
worst_case_index = indices.pop(0)

combinations = []
# generate 1000000 combinations
for _ in range(500000):
    c = random.sample(indices, 3)
    c.append(worst_case_index)
    combinations.append(tuple(c))

# Normalize outcomes
x = combined_df[outcomes_of_interest].values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
normalized_outcomes = pd.DataFrame(x_scaled, columns=outcomes_of_interest)

# Reset index and create a mapping
# mapping = {}
# reverse = {}
# for idx, row in combined_df.iterrows():
#     mapping[row['Run ID']] = idx
#     reverse[idx] = row['Run ID']

# # Map combinations
# combinations_in_normalized_outcomes = []
# for c in combinations:
#     mapped_c = [mapping[x] for x in c]
#     combinations_in_normalized_outcomes.append(tuple(mapped_c))

# calculate the pairwise distances between the normalized outcomes
distances = squareform(pdist(normalized_outcomes.values))

from multiprocessing import Process, Manager

def worker(id, distances, combinations, return_dict):
    return_dict[id] = find_maxdiverse_scenarios(distances, combinations)

manager = Manager()
return_dict = manager.dict()
processes = []
cores = os.cpu_count()
for i in range(cores):
    p_data = np.array_split(combinations, cores)[i]
    p = Process(target=worker, args=(i, distances, p_data, return_dict))
    processes.append(p)
    print('starting process', i)
    p.start()

for i, p in enumerate(processes):
    p.join()
    print('joined process', i)

# print(return_dict)
# partial_function = functools.partial(find_maxdiverse_scenarios, distances)

# print('before parallel execution')
# # setup the pool of workers and split the calculations over the set of cores
# with ProcessPoolExecutor(max_workers=cores) as executor:
#     worker_data = np.array_split(combinations_in_normalized_outcomes, cores)
#     diversity_results = [e for e in executor.map(partial_function, worker_data)]
#     diversity_results = list(itertools.chain.from_iterable(diversity_results))
#     print('after parallel execution')

results_list = []
for id, results in return_dict.items():
    for result in results:
        score = result[0][0]
        combination = list(result[1])
        results_list.append({'score':score,
                             'combination':combination})

# diversity_results = np.concatenate(return_list)

# Capture results
results_list.sort(key=lambda entry:entry['score'], reverse=True)

most_diverse = results_list[0]
most_diverse_set = most_diverse['combination']
print(most_diverse_set)

selected = combined_df.loc[most_diverse['combination'], uncertainties]
scenarios = [Scenario(f"{index}", **row) for index, row in selected.iterrows()]
for scenario in scenarios:
    print(scenario)

selected_forsave = combined_df.loc[most_diverse['combination'], ['Run ID'] + uncertainties]
selected_forsave.to_csv('output/selected_scenarios.csv', index=False)