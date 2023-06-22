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

    # Generally, we are attempting to find policies that serve our local
    # community (Dike Ring 4) while also serving the IJssel region as a whole.
    # Our clients, representatives of the Dike Ring, seek a policy that is
    # maximally beneficial to their direct needs and wants. At the same time,
    # we seek to propose a policy that serves the whole region, to help our
    # clients gain favour at the negotiation table.
    #
    # Our client's mandate centers around minimizing harm in Dike Ring 4.
    # Beyond that, they seek to ensure that other Dike Rings do not see
    # disproportionately more protection than they do, and that they are not
    # disproportionately more burdened by infrastructure projects and land
    # reclamation than the other regions.
    #
    # Thus, we define two main "problem formulations" which we will run inside
    # the model. The first ('A4 Only') focuses just on local effects (deaths
    # and amages). The second ('All Dikes') includes the deaths and damages in
    # each dike ring. Both include "totals" (sums of deaths/damages across all
    # dike rings): A4 Only does so so that, when we use it in service of policy
    # optimization, we do not neglect total damages in our search; All Dikes
    # includes this outcome simply for convenience, as it is duplicated by its
    # other outcomes. Both formulations also include a 'Total Infrasutructure
    # Costs' outcome, that is the sum of the cost of Dike Heightening and RfR
    # measures. Who pays for what is fundamentally a political question tied to
    # numerous factors including how effective a proposed policy is in 
    # protecting each region. Thus, it would be dishonest to optimize for one
    # particular type of cost, unless we were explicitly trying to avoid a 
    # particular type of policy, which we are not. The outputs of our modelling
    # can be used down the line to justify a particular cost sharing scheme, if
    # politically necessary. 
    #
    # In some parts of our analysis, we consider other important outcomes or
    # metrics, such as the ratio of damages in Dike Ring 4 to those in
    # Dike Rings 1 & 2 (the regions of industrial farmers). Since these are
    # explicitly calculable from our full set of outcomes, and since we do not
    # use them in optimization (as that would produce policies that would not
    # see political consensus), we calculate them in our post-hoc analysis.

    # Outcomes are all costs, thus they have to minimized:
    direction = ScalarOutcome.MINIMIZE

    if problem_formulation_id == 'A4 Only':

        outcomes = []

        # A4 Deaths and Damages
        for dike in function.dikelist:
            for entry in [
                "Expected Annual Damage",
                "Expected Number of Deaths",
            ]:
                outcome_name = ''.join(dike.split('.'))
                outcome_name += '_' + '_'.join(entry.split(' '))
                if dike == "A.4":
                    outcomes.append(
                        ScalarOutcome(
                            outcome_name,
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

        # All Dike Ring Deaths and Damages
        for dike in function.dikelist:
            for entry in [
                "Expected Annual Damage",
                "Expected Number of Deaths",
            ]:
                outcome_name = ''.join(dike.split('.'))
                outcome_name += '_' + '_'.join(entry.split(' '))

                # We mark A.4 with "kind=Outcome.MINIMIZE" so that this problem
                # formulation can be used in an optimization without expliticly
                # optimizing for all 13 outcomes. Due to some nuances in the
                # ema_workbench, this was ultimately unused.

                if dike == "A.4":
                    outcomes.append(
                        ScalarOutcome(
                            outcome_name,
                            variable_name=f"{dike}_{entry}",
                            function=sum_over,
                            kind=direction,
                        )
                    )
                else:
                    outcomes.append(
                        ScalarOutcome(
                            outcome_name,
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

        # Aggregated Deaths and Damages (for convenience)
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
