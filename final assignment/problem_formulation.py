"""
Created on Wed Mar 21 17:34:11 2018

@author: ciullo
"""
from ema_workbench import (
    Model,
    CategoricalParameter,
    ArrayOutcome,
    ScalarOutcome,
    IntegerParameter,
    RealParameter,
)
from dike_model_function import DikeNetwork  # @UnresolvedImport

import numpy as np


def sum_over(*args):
    numbers = []
    for entry in args:
        try:
            value = sum(entry)
        except TypeError:
            value = entry
        numbers.append(value)

    return sum(numbers)


def sum_over_time(*args):
    data = np.asarray(args)
    summed = data.sum(axis=0)
    return summed


def get_model_for_problem_formulation(problem_formulation_id):
    """Convenience function to prepare DikeNetwork in a way it can be input in the EMA-workbench.
    Specify uncertainties, levers, and outcomes of interest.

    Parameters
    ----------
    problem_formulation_id : int {0, ..., 7}
                             problem formulations differ with respect to the objectives
                             DEFAULT:
                             0: Total cost, and casualties
                             1: Expected damages, costs, and casualties
                             2: expected damages, dike investment costs, rfr costs, evacuation cost, and casualties
                             3: costs and casualties disaggregated over dike rings, and room for the river and evacuation costs
                             4: Expected damages, dike investment cost and casualties disaggregated over dike rings and room for the river and evacuation costs
                             5: disaggregate over time and space
                             CUSTOM:
                             6:
                             7:

    Notes
    -----
    problem formulations 4 and 5 rely on ArrayOutcomes and thus cannot straightforwardly
    be used in optimizations

    """
    
    # Load the model:
    function = DikeNetwork()
    # workbench model:
    dike_model = Model("dikesnet", function=function)

    dike_model.uncertainties = [
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

    dike_model.levers = [
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

    # Problem formulations:
    # Outcomes are all costs, thus they have to minimized:
    direction = ScalarOutcome.MINIMIZE

    # OUTCOMES OF INTEREST:
    # In Open Exploration, we care about (5 total):
    #   Deaths to DR4
    #   Damages in DR4
    #   Total infrastructure costs (all DRs)
    #   Total deaths (all DRs)
    #   Total damages (all DRs)
    # In DS, we care about (13 total) (some will be set kind=info so they are ignored by optimizer):
    #   Deaths per DR (DRs 1, 2, 3, 5 := info)
    #   Damages per DR (DRs 1, 2, 3, 5 := info)
    #   Total infrastructure costs (all DRs)
    #   Total deaths (all DRs)
    #   Total damages (all DRs)
    # Thus, we can assemble one Problem Formulation that includes all of the
    #  above and set kind=INFO where relevant (kind defaults to INFO) for DS
    #  and simply drop those columns from the outputs for OE as desired

    if problem_formulation_id == 'A4 Only':

        outcomes = []
        # A4 Deaths and Damages
        for dike in function.dikelist:
            for entry in [
                "Expected Annual Damage",
                "Expected Number of Deaths",
            ]:
                unc_name = ''.join(dike.split('.'))
                unc_name += '_' + '_'.join(entry.split(' '))
                if dike == "A.4":
                    outcomes.append(
                        ScalarOutcome(
                            unc_name,
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

        dike_model.outcomes = outcomes

    elif problem_formulation_id == 'All Dikes':
        outcomes = []
        # A4 Deaths and Damages
        for dike in function.dikelist:
            for entry in [
                "Expected Annual Damage",
                "Expected Number of Deaths",
            ]:
                unc_name = ''.join(dike.split('.'))
                unc_name += '_'.join(entry.split(' '))
                if dike == "A.4":
                    outcomes.append(
                        ScalarOutcome(
                            unc_name,
                            variable_name=f"{dike}_{entry}",
                            function=sum_over,
                            kind=direction,
                        )
                    )
                else:
                    outcomes.append(
                        ScalarOutcome(
                            unc_name,
                            variable_name=f"{dike}_{entry}",
                            function=sum_over,
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

        dike_model.outcomes = outcomes

    else:
        raise TypeError("unknown identifier")

    return dike_model, function.planning_steps



if __name__ == "__main__":
    get_model_for_problem_formulation(3)
