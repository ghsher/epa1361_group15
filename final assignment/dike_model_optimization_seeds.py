from ema_workbench import (
    Model,
    MultiprocessingEvaluator,
    ScalarOutcome,
    IntegerParameter,
    optimize,
    Scenario,
)
from ema_workbench.em_framework.optimization import EpsilonProgress
from ema_workbench.em_framework.outcomes import AbstractOutcome
from ema_workbench.util import ema_logging

from problem_formulation import get_model_for_problem_formulation
import matplotlib.pyplot as plt
import seaborn as sns
from ema_workbench.em_framework.optimization import ArchiveLogger, EpsilonProgress
from datetime import datetime
import pandas as pd

if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    model, steps = get_model_for_problem_formulation(7)

    reference_values = {
        "Bmax": 175,
        "Brate": 1.5,
        "pfail": 0.5,
        "discount rate 0": 3.5,
        "discount rate 1": 3.5,
        "discount rate 2": 3.5,
        "ID flood wave shape": 4,
    }
    scen1 = {}

    for key in model.uncertainties:
        name_split = key.name.split("_")

        if len(name_split) == 1:
            scen1.update({key.name: reference_values[key.name]})

        else:
            scen1.update({key.name: reference_values[name_split[1]]})

    ref_scenario = Scenario("reference", **scen1)

    # espilon = [0.1] * len([outcome for outcome in model.outcomes if outcome.kind != AbstractOutcome.INFO])
    # we decide to stick with the below, due to example outcome values generated in some optimization:
    # 4185255.822441114,0.00032990200518328536,135802735.00849408,30559778.753226846,0.004074074031800072
    espilon = [100, 0.01, 100, 100, 0.01]

    # nfe = 100  # proof of principle only, way to low for actual use
    # nfe = 40000 <- this is what Agnes will run on 14.06, should take arounf 5h
    nfe = 400 # <- this is good for test runs <5min
    # nfe's in Nicolo's Paper: 100K, Paper Fotini: 40K

    # we need to store our results for each seed
    results = []
    convergences = []

    CURRENT_DATE = datetime.today().strftime('%Y%m%dT%H%M')

    with MultiprocessingEvaluator(model) as evaluator:
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
                    base_filename=CURRENT_DATE + f"_{i}" +".tar.gz",
                ),
                EpsilonProgress(),
            ]

            result, convergence = evaluator.optimize(
                nfe=nfe,
                searchover="levers",
                epsilons=espilon,
                convergence=convergence_metrics,
                reference=ref_scenario,
            )
            results.append(result)
            convergences.append(convergence)

    convergence_seed = pd.concat(convergences, ignore_index=False) #agnes
    result_seed = pd.concat(results, ignore_index=False) #agnes

    result_seed.to_csv('./output/' + CURRENT_DATE + '_results_seed_ds.csv')
    convergence_seed.to_csv('./output/' + CURRENT_DATE + '_convergence_seed_ds.csv')

    #from ema_workbench.em_framework.optimization import epsilon_nondominated, to_problem
    #problem = to_problem(model, searchover="levers")
    #print(problem.parameter_names)
    #merged_archives = epsilon_nondominated(result_seed, espilon, problem) #agnes
    #merged_archives.to_csv('./output/' + CURRENT_DATE + '_merged_archives_seed_ds.csv')

    #this plots the epsilon convergence from the last seed, so not very useful
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True)
    fig, ax1 = plt.subplots(ncols=1)
    ax1.plot(convergence.epsilon_progress)
    ax1.set_xlabel("nr. of generations")
    ax1.set_ylabel(r"$\epsilon$ progress")
    sns.despine()
    plt.show()
