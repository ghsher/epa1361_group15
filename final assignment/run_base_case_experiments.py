from ema_workbench import (
    Policy,
    ema_logging,
    MultiprocessingEvaluator,
    save_results,
    Samplers
)
from dike_model_function import DikeNetwork  # @UnresolvedImport
from problem_formulation import get_model_for_problem_formulation

ema_logging.log_to_stderr(ema_logging.INFO)

# choose problem formulation number, between 0-5
# each problem formulation has its own list of outcomes
dike_model, planning_steps = get_model_for_problem_formulation('All Dikes')

def get_do_nothing_dict():
    return {l.name: 0 for l in dike_model.levers}

policies = []

# Do Nothing case
policies.append(Policy("Base Case", **dict(get_do_nothing_dict(),)))

# pass the policies list to EMA workbench experiment runs
N = 100000
with MultiprocessingEvaluator(dike_model) as evaluator:
    results = evaluator.perform_experiments(N, policies)
    
filename = './output/' + 'base_case' + '_results__' + str(N) + '_scenarios.tar.gz'
save_results(results, filename)