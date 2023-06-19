#TODO remove unnecessary imports

# from ema_workbench import (
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
from ema_workbench import MultiprocessingEvaluator, ema_logging
from ema_workbench.em_framework.evaluators import BaseEvaluator

from ema_workbench.em_framework.optimization import (ArchiveLogger,
                                                     EpsilonProgress,
                                                     to_problem, epsilon_nondominated)

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

    epsilons = [100, 0.01, 100, 100, 0.01]

    # TODO, run with 30K
    # nfe = 30000
    nfe = 400

    CURRENT_DATE = datetime.today().strftime('%Y%m%dT%H%M')
    scenarios=[]
    # TODO, pass real scenarios
    scenarios.append(ref_scenario) #agnes

    def optimize(scenario, nfe, model, epsilons):
        results = []
        convergences = []
        problem = to_problem(model, searchover="levers")

        with MultiprocessingEvaluator(model) as evaluator:
            for i in range(5):
                convergence_metrics = [
                    ArchiveLogger(
                        "./archives",
                        [l.name for l in model.levers],
                        [o.name for o in model.outcomes],
                        base_filename = CURRENT_DATE + f"_{scenario.name}_seed_{i}.tar.gz",
                    ),
                    EpsilonProgress(),
                ]

                result, convergence = evaluator.optimize(nfe=nfe, searchover='levers',
                                                         convergence=convergence_metrics,
                                                         epsilons=epsilons,
                                                         reference=scenario)

                results.append(result)
                convergences.append(convergence)
                result.to_csv('./output/' + CURRENT_DATE + f"result_{scenario.name}_seed_{i}.csv")
                convergence.to_csv('./output/' + CURRENT_DATE + f"convergence_{scenario.name}_seed_{i}.csv")

        # merge the results using a non-dominated sort.
        # reference_set = epsilon_nondominated(results, epsilons, problem)
        # TODO: since the solution fails due to " " in names, hardcode to see if script runs
        reference_set = results[0]

        return reference_set, convergences

    results = []
    for scenario in scenarios:
        results.append(optimize(scenario, nfe, model, epsilons))

