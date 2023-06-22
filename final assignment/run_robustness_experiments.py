from ema_workbench import (
    Policy,
    ema_logging,
    MultiprocessingEvaluator,
    save_results,
)
from dike_model_function import DikeNetwork  # @UnresolvedImport
from problem_formulation import get_model_for_problem_formulation
import pandas as pd

ema_logging.log_to_stderr(ema_logging.INFO)

dike_model, planning_steps = get_model_for_problem_formulation('A4 Only')

policies_df = pd.read_csv('./output/policies__constraints_filtered__diverse_set_50.csv',
                       index_col='Policy Name')
policies_df = policies_df[[L.name for L in dike_model.levers]]

policies = []
for name, policy in policies_df.iterrows():
    policies.append(Policy(str(name), **policy.to_dict()))

# N scenarios
N = 1000
with MultiprocessingEvaluator(dike_model) as evaluator:
    results = evaluator.perform_experiments(N, policies)

filename = './output/' + 'robustness' + '_results__' + str(N) + '_scenarios.tar.gz'
save_results(results, filename)