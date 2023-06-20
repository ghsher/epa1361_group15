import random
import pandas as pd
from sklearn import preprocessing
import os
from scipy.spatial.distance import pdist, squareform
from scenario_diversity import find_maxdiverse_scenarios
import numpy as np
from ema_workbench import Scenario

random.seed(1361)

combined_df = pd.read_csv('output/base_case_results__100000_scenarios__combined_df.csv')
combined_df = combined_df.rename({'Unnamed: 0' : 'Run ID'}, axis=1)

## ASSEMBLE 1M SCENARIO COMBINATIONS 

# Get a list of all incides in the DataFrame
indices = []
for idx, row in combined_df.iterrows():
    indices.append(idx)
worst_case_index = indices.pop(0)

# Randomly generate sets
combinations = []
for _ in range(1000000):
    c = random.sample(indices, 3)
    c.append(worst_case_index)
    combinations.append(tuple(c))

## ASSESS DIVERSITY OF EACH COMBINATION

# Select and rename columns of interest
combined_df['Dike Rings 1 & 2 Damage/Year'] =       \
        combined_df['A.1 Expected Annual Damage'] + \
        combined_df['A.2 Expected Annual Damage']

combined_df = combined_df.rename({
    'A.4 Expected Annual Damage'    : 'Dike Ring 4 Damage/Year',
    'Total Expected Annual Damage'  : 'Total Damage/Year',
}, axis=1)

outcomes_of_interest = ['Dike Rings 1 & 2 Damage/Year',
                        'Dike Ring 4 Damage/Year',
                        'Total Damage/Year',]
outcomes_df = combined_df[outcomes_of_interest].values

# Scale (normalize) outcome data
min_max_scaler = preprocessing.MinMaxScaler()
outcomes_scaled = min_max_scaler.fit_transform(outcomes_df)
normalized_outcomes = pd.DataFrame(outcomes_scaled, columns=outcomes_of_interest)

# Calculate the pairwise distances between the normalized outcomes
distances = squareform(pdist(normalized_outcomes.values))


# Split up the diversity-calculating task between processes

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


## ASSESS DIVERSITY SCORES

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

uncertainties = ['A.0_ID flood wave shape', 'A.1_Bmax', 'A.1_Brate', 'A.1_pfail', 'A.2_Bmax',
                 'A.2_Brate', 'A.2_pfail', 'A.3_Bmax', 'A.3_Brate', 'A.3_pfail', 'A.4_Bmax',
                 'A.4_Brate', 'A.4_pfail', 'A.5_Bmax', 'A.5_Brate', 'A.5_pfail',
                 'discount rate 0', 'discount rate 1', 'discount rate 2',]

selected = combined_df.loc[most_diverse['combination'], uncertainties]
scenarios = [Scenario(f"{index}", **row) for index, row in selected.iterrows()]
for scenario in scenarios:
    print(scenario)

selected_forsave = combined_df.loc[most_diverse['combination'], ['Run ID'] + uncertainties]
selected_forsave.to_csv('output/selected_scenarios.csv', index=False)