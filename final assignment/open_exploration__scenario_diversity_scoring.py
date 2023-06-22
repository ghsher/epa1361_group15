import random
import pandas as pd
from sklearn import preprocessing
import os
from scipy.spatial.distance import pdist, squareform
from scenario_diversity import find_maxdiverse
import numpy as np
from ema_workbench import Scenario
from ema_workbench.analysis import parcoords
import matplotlib.pyplot as plt
import seaborn as sns

random.seed(1361)

combined_df = pd.read_csv('output/base_case_results__100000_scenarios__prim_filtered.csv')
combined_df = combined_df.rename({'Unnamed: 0' : 'Run ID'}, axis=1)
# combined_df = combined_df.set_index('Run ID')

## ASSEMBLE 1M SCENARIO COMBINATIONS 

# Get a list of all incides in the DataFrame
indices = []
for idx, row in combined_df.iterrows():
    indices.append(idx)
worst_case_index = indices.pop(0)

# Randomly generate sets
combinations = []
for _ in range(2000000):
    c = random.sample(indices, 3)
    c.append(worst_case_index)
    combinations.append(tuple(c))

## ASSESS DIVERSITY OF EACH COMBINATION

# Select and rename columns of interest
combined_df['Dike Rings 1 & 2 Damage/Year'] =       \
        combined_df['A1_Expected_Annual_Damage'] + \
        combined_df['A2_Expected_Annual_Damage']

combined_df = combined_df.rename({
    'A4_Expected_Annual_Damage'    : 'Dike Ring 4 Damage/Year',
    'Total_Expected_Annual_Damage'  : 'Total Damage/Year',
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
    return_dict[id] = find_maxdiverse(distances, combinations)

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

# Capture results
results_list.sort(key=lambda entry:entry['score'], reverse=True)

most_diverse = results_list[0]
most_diverse_set = most_diverse['combination']
run_ids = [combined_df.loc[s, 'Run ID'] for s in most_diverse_set]
print("Selected most diverse set: Run IDs", run_ids)

uncertainties = ['A0_ID_flood_wave_shape', 'A1_Bmax', 'A1_Brate', 'A1_pfail', 'A2_Bmax',
                 'A2_Brate', 'A2_pfail', 'A3_Bmax', 'A3_Brate', 'A3_pfail', 'A4_Bmax',
                 'A4_Brate', 'A4_pfail', 'A5_Bmax', 'A5_Brate', 'A5_pfail',
                 'discount_rate_0', 'discount_rate_1', 'discount_rate_2',]

selected = combined_df.loc[most_diverse_set, ['Run ID'] + uncertainties]
selected.to_csv('output/selected_scenarios.csv', index=False)

## PLOTTING

# Overwrite outcomes_df to have headers
outcomes_df = combined_df[outcomes_of_interest]

limits = parcoords.get_limits(outcomes_df)
axes = parcoords.ParallelAxes(limits)

# we set the linewidth lower, and make the lines slightly transpartant using alpha
# this often helps reveal patterns in the results.
axes.plot(outcomes_df, color='lightgrey', lw=0.5, alpha=0.5)
for i, scenario in enumerate(most_diverse_set):
    axes.plot(outcomes_df.loc[scenario, :], color=sns.color_palette()[i], lw=1)

fig = plt.gcf()
fig.set_size_inches((12, 20))

plt.savefig('img/selected_scenarios__outcome_plot.png')
plt.show()