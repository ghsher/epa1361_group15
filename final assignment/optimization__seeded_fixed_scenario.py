from ema_workbench import (
    MultiprocessingEvaluator,
    Scenario,
)

from ema_workbench.em_framework.optimization import EpsilonProgress, ArchiveLogger
from ema_workbench.util import ema_logging
from ema_workbench.em_framework.optimization import epsilon_nondominated, to_problem

import pandas as pd
from problem_formulation import get_model_for_problem_formulation


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    model, steps = get_model_for_problem_formulation('All Dikes')
    problem = to_problem(model, searchover='levers')

    reference_values = {
        "Bmax": 175,
        "Brate": 1.5,
        "pfail": 0.5,
        "discount_rate_0": 3.5,
        "discount_rate_1": 3.5,
        "discount_rate_2": 3.5,
        "ID_flood_wave_shape": 4,
    }

    scenarios_df = pd.read_csv('output/selected_scenarios.csv')
    print(scenarios_df)
    SCENARIO_NUMBERS = [0]#, 1, 2, 3, 4] # 0, 1, 2, 3, or 4 (4 == reference)
    
    scenarios = []
    for id in SCENARIO_NUMBERS:
        scen = {}
        if id == 4:
            for key in model.uncertainties:
                key_name_split = key.name.split("_")
                dike = key_name_split[0]
                if dike == 'A0':
                    scen.update({key.name : reference_values['ID_flood_wave_shape']})
                elif dike[0] == 'A':
                    scen.update({key.name : reference_values[key_name_split[1]]})
                else:
                    scen.update({key.name : reference_values[key.name]})
            scenario = Scenario("Reference", **scen)
            
        else:
            for col in scenarios_df:
                if col == 'Run ID':
                    continue
                scen.update({col : scenarios_df.loc[id, col]})
            scenario = Scenario(scenarios_df.loc[id, 'Run ID'], **scen)

        scenarios.append(scenario)

    espilon = [100, 0.01, 100, 100, 0.01]

    nfe = 30000

    # we need to store our results for each seed
    results = []
    convergences = []

    with MultiprocessingEvaluator(model) as evaluator:
        for scenario in scenarios:
            # we run again for 5 seeds
            for i in range(5):
                # we create 2 covergence tracker metrics
                # the archive logger writes the archive to disk for every x nfe
                # the epsilon progress tracks during runtime
                convergence_metrics = [
                    ArchiveLogger(
                        "./archives",
                        [l.name for l in model.levers],
                        [o.name for o in model.outcomes],
                        base_filename=f"POLICY_SEARCH__archive__scen{scenario.name}__seed{i}.tar.gz"
                    ),
                    EpsilonProgress(),
                ]

                result, convergence = evaluator.optimize(
                    nfe=nfe,
                    searchover="levers",
                    epsilons=espilon,
                    convergence=convergence_metrics,
                    reference=scenario,
                )
                results.append(result)
                convergences.append(convergence)

                filename_start = './output/POLICY_SEARCH__'
                filename_end = f'__scen{scenario.name}__seed{i}.csv'
                result.to_csv(filename_start + 'results' + filename_end)
                convergence.to_csv(filename_start + 'convergence' + filename_end)
        
            scenario_results = results[-5:]
            merged_policies = epsilon_nondominated(scenario_results, espilon, problem)
            merged_policies.to_csv(f'./output/POLICY_SEARCH__merged__scen{scenario}.csv')