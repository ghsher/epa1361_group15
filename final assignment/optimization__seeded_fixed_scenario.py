from ema_workbench import (
    Model,
    MultiprocessingEvaluator,
    ScalarOutcome,
    IntegerParameter,
    optimize,
    Scenario,
    CategoricalParameter,
    ArrayOutcome,
    RealParameter,
)

from ema_workbench.em_framework.optimization import EpsilonProgress, ArchiveLogger
from ema_workbench.em_framework.outcomes import AbstractOutcome
from ema_workbench.util import ema_logging
from ema_workbench.em_framework.optimization import epsilon_nondominated, to_problem

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
from dike_model_function import DikeNetwork  # @UnresolvedImport

def sum_over(*args):
    numbers = []
    for entry in args:
        try:
            value = sum(entry)
        except TypeError:
            value = entry
        numbers.append(value)

    return sum(numbers)


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    # model, steps = get_model_for_problem_formulation(7)

### MODEL ASSEMBLY STARTING ###

    # Load the model:
    function = DikeNetwork()
    model = Model("dikesnet", function=function)

    model.uncertainties = [
        RealParameter("A1_Bmax", 30, 350, variable_name="A.1_Bmax"),
        RealParameter("A1_pfail", 0, 1, variable_name="A.1_pfail"),
        CategoricalParameter("A1_Brate", [1.0, 1.5, 10], variable_name="A.1_Brate"),
        RealParameter("A2_Bmax", 30, 350, variable_name="A.2_Bmax"),
        RealParameter("A2_pfail", 0, 1, variable_name="A.2_pfail"),
        CategoricalParameter("A2_Brate", [1.0, 1.5, 10], variable_name="A.2_Brate"),
        RealParameter("A3_Bmax", 30, 350, variable_name="A.3_Bmax"),
        RealParameter("A3_pfail", 0, 1, variable_name="A.3_pfail"),
        CategoricalParameter("A3_Brate", [1.0, 1.5, 10], variable_name="A.3_Brate"),
        RealParameter("A4_Bmax", 30, 350, variable_name="A.4_Bmax"),
        RealParameter("A4_pfail", 0, 1, variable_name="A.4_pfail"),
        CategoricalParameter("A4_Brate", [1.0, 1.5, 10], variable_name="A.4_Brate"),
        RealParameter("A5_Bmax", 30, 350, variable_name="A.5_Bmax"),
        RealParameter("A5_pfail", 0, 1, variable_name="A.5_pfail"),
        CategoricalParameter("A5_Brate", [1.0, 1.5, 10], variable_name="A.5_Brate"),
        CategoricalParameter("discount_rate_0", (1.5, 2.5, 3.5, 4.5), variable_name="discount rate 0"),
        CategoricalParameter("discount_rate_1", (1.5, 2.5, 3.5, 4.5), variable_name="discount rate 1"),
        CategoricalParameter("discount_rate_2", (1.5, 2.5, 3.5, 4.5), variable_name="discount rate 2"),
        IntegerParameter("A0_ID_flood_wave_shape", 0, 132, variable_name="A.0_ID flood wave shape"),
    ]

    model.levers = [
        IntegerParameter("EWS_DaysToThreat", 0, 4),
        IntegerParameter("rfr_0_t0", 0, 1, variable_name="0_RfR 0"),
        IntegerParameter("rfr_0_t1", 0, 1, variable_name="0_RfR 1"),
        IntegerParameter("rfr_0_t2", 0, 1, variable_name="0_RfR 2"),
        IntegerParameter("rfr_1_t0", 0, 1, variable_name="1_RfR 0"),
        IntegerParameter("rfr_1_t1", 0, 1, variable_name="1_RfR 1"),
        IntegerParameter("rfr_1_t2", 0, 1, variable_name="1_RfR 2"),
        IntegerParameter("rfr_2_t0", 0, 1, variable_name="2_RfR 0"),
        IntegerParameter("rfr_2_t1", 0, 1, variable_name="2_RfR 1"),
        IntegerParameter("rfr_2_t2", 0, 1, variable_name="2_RfR 2"),
        IntegerParameter("rfr_3_t0", 0, 1, variable_name="3_RfR 0"),
        IntegerParameter("rfr_3_t1", 0, 1, variable_name="3_RfR 1"),
        IntegerParameter("rfr_3_t2", 0, 1, variable_name="3_RfR 2"),
        IntegerParameter("rfr_4_t0", 0, 1, variable_name="4_RfR 0"),
        IntegerParameter("rfr_4_t1", 0, 1, variable_name="4_RfR 1"),
        IntegerParameter("rfr_4_t2", 0, 1, variable_name="4_RfR 2"),
        IntegerParameter(
            "A1_DikeIncrease_t0", 0, 10, variable_name="A.1_DikeIncrease 0"
        ),
        IntegerParameter(
            "A1_DikeIncrease_t1", 0, 10, variable_name="A.1_DikeIncrease 1"
        ),
        IntegerParameter(
            "A1_DikeIncrease_t2", 0, 10, variable_name="A.1_DikeIncrease 2"
        ),
        IntegerParameter(
            "A2_DikeIncrease_t0", 0, 10, variable_name="A.2_DikeIncrease 0"
        ),
        IntegerParameter(
            "A2_DikeIncrease_t1", 0, 10, variable_name="A.2_DikeIncrease 1"
        ),
        IntegerParameter(
            "A2_DikeIncrease_t2", 0, 10, variable_name="A.2_DikeIncrease 2"
        ),
        IntegerParameter(
            "A3_DikeIncrease_t0", 0, 10, variable_name="A.3_DikeIncrease 0"
        ),
        IntegerParameter(
            "A3_DikeIncrease_t1", 0, 10, variable_name="A.3_DikeIncrease 1"
        ),
        IntegerParameter(
            "A3_DikeIncrease_t2", 0, 10, variable_name="A.3_DikeIncrease 2"
        ),
        IntegerParameter(
            "A4_DikeIncrease_t0", 0, 10, variable_name="A.4_DikeIncrease 0"
        ),
        IntegerParameter(
            "A4_DikeIncrease_t1", 0, 10, variable_name="A.4_DikeIncrease 1"
        ),
        IntegerParameter(
            "A4_DikeIncrease_t2", 0, 10, variable_name="A.4_DikeIncrease 2"
        ),
        IntegerParameter(
            "A5_DikeIncrease_t0", 0, 10, variable_name="A.5_DikeIncrease 0"
        ),
        IntegerParameter(
            "A5_DikeIncrease_t1", 0, 10, variable_name="A.5_DikeIncrease 1"
        ),
        IntegerParameter(
            "A5_DikeIncrease_t2", 0, 10, variable_name="A.5_DikeIncrease 2"
        ),
    ]

    direction = ScalarOutcome.MINIMIZE
    outcomes = []

    # Disaggregated Deaths and Damages
    for dike in function.dikelist:
        for entry in [
            "Expected_Annual_Damage",
            "Expected_Number_of_Deaths",
        ]:
            if dike == "A.4":
                outcomes.append(
                    ScalarOutcome(
                        f"A4_{entry}",
                        variable_name=f"{dike}_{entry}",
                        function=sum_over,
                        kind=direction,
                    )
                )

    # Aggregated Costs
    cost_variables = []
    cost_variables.extend(
        [f"{dike}_Dike Investment Costs" for dike in function.dikelist]
    )
    cost_variables.extend([f"RfR Total Costs"])

    outcomes.append(
        ScalarOutcome(
            "Total_Infrastructure_Costs",
            variable_name=[var for var in cost_variables],
            function=sum_over,
            kind=direction,
        )
    )

    # Aggregated Deaths and Damages
    total_damage_variables = []
    total_damage_variables.extend(
        [f"{dike}_Expected Annual Damage" for dike in function.dikelist]
    )

    total_casualty_variables = []
    total_casualty_variables.extend(
        [f"{dike}_Expected Number of Deaths" for dike in function.dikelist]
    )

    outcomes.append(
        ScalarOutcome(
            "Total_Expected_Annual_Damage",
            variable_name=[var for var in total_damage_variables],
            function=sum_over,
            kind=direction,
        )
    )
    outcomes.append(
        ScalarOutcome(
            "Total_Expected_Number_of_Deaths",
            variable_name=[var for var in total_casualty_variables],
            function=sum_over,
            kind=direction,
        )
    )

    model.outcomes = outcomes
    problem = to_problem(model, searchover="levers")

### MODEL ASSEMBLY DONE ###

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
    SCENARIO_NUMBERS = [0, 1, 2, 3, 4] # 0, 1, 2, 3, or 4 (4 == reference)
    
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
                        base_filename=f"DIRECTED_SEARCH__archive__scen{scenario.name}__seed{i}.tar.gz"
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

                filename_start = './output/DIRECTED_SEARCH__'
                filename_end = f'__scen{scenario.name}__seed{i}.csv'
                result.to_csv(filename_start + 'results' + filename_end)
                convergence.to_csv(filename_start + 'convergence' + filename_end)

        merged_archives = epsilon_nondominated(results, espilon, problem)
        merged_archives.to_csv(f'./output/DIRECTED_SEARCH__merged_archives__scen{scenario}.csv')