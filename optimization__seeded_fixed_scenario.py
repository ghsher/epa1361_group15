import pandas as pd

from ema_workbench import (
    MultiprocessingEvaluator,
    Scenario,
)
from ema_workbench.em_framework.optimization import (
    EpsilonProgress,
    ArchiveLogger,
    epsilon_nondominated,
    to_problem,
)
from ema_workbench.util import ema_logging

from problem_formulation import get_model_for_problem_formulation


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    model, steps = get_model_for_problem_formulation('A4 Only')
    problem = to_problem(model, searchover='levers')

    scenarios_df = pd.read_csv('output/selected_scenarios.csv')
    print(scenarios_df)
    
    scenarios = []
    for id in range(scenarios_df.shape[0]):
        scen = {}
        for col in scenarios_df:
            if col == 'Run ID':
                continue
            scen.update({col : scenarios_df.loc[id, col]})

        ema_scenario = Scenario(scenarios_df.loc[id, 'Run ID'], **scen)
        scenarios.append(ema_scenario)

    espilon = [100, 0.01, 100, 100, 0.01]

    nfe = 20000

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