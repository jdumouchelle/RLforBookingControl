import argparse
import pickle
import subprocess

import rm.params as params
from rm.utils import factory_get_path



#--------------------------------------------------------#
#                 Functions to configs                   #
#--------------------------------------------------------#

class A(object):
    pass 


def get_problem_from_param():
    """ Gets all problems form params"""
    def is_problem(p):
        if "cm_" in p or "vrp_" in p:
            return True
        return False

    problems = list(filter(lambda x: is_problem(x), dir(params)))
    return problems


def get_results_for_problem(args, problem):
    """ Loads results for specific problem.  """
    # get a dummy class with A.problem = "problem"
    tmp_args = A()
    tmp_args.problem = problem
    get_path = factory_get_path(tmp_args)

    # load data from corresponding problem
    cfg = getattr(params, problem)
    fp = get_path(cfg, "combined_results", args.data_dir)
    with open(fp, 'rb') as p:
        problem_results = pickle.load(p)

    return problem_results



#--------------------------------------------------------#
#                          Main                          #
#--------------------------------------------------------#

def main(args):
    # Get the names of problems from params.py
    problems = get_problem_from_param()

    combined_results = {}
    for problem in problems:

        # collect results for instance
        cmd_as_list = ["python", "-m", "rm.scripts.collect_results",
                       "--problem", problem,
                       "--data_dir", args.data_dir]

        subprocess.call(cmd_as_list)
        combined_results[problem] = get_results_for_problem(args, problem)

    # store results
    fp_res = args.data_dir + '/results.pkl'
    with open(fp_res, 'wb') as p:
        pickle.dump(combined_results, p)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collects are stores all results.')

    parser.add_argument('--data_dir', type=str, default="./data/")
    args = parser.parse_args()

    main(args)
