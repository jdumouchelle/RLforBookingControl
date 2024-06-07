import pickle


def factory_get_path(args):
    """ Gets factory get_path. """
    if 'vrp' in args.problem:
        from .vrp import get_path
        return get_path

    elif 'cm' in args.problem:
        from .cm import get_path
        return get_path

    else:
        raise Exception(f"rm.utils not defined for problem class {args.problem}")


def factory_env(args):
    """ Gets factory Environment. """
    if 'vrp' in args.problem:
        from rm.envs.vrp.environment import Environment
        return Environment

    elif 'cm' in args.problem:
        from rm.envs.cm.environment import Environment
        return Environment

    else:
        raise Exception(f"Not a valid problem: {args.problem}")


def factory_gym_env(args):
    """ Gets factory GymEnvironment. """
    if 'vrp' in args.problem:
        from rm.envs.vrp.rl_environment import GymEnvironment
        return GymEnvironment

    elif 'cm' in args.problem:
        from rm.envs.cm.rl_environment import GymEnvironment
        return GymEnvironment

    else:
        raise Exception(f"Not a valid problem: {args.problem}")


def factory_data_generator(args):
    """ Gets factory DataGenerator. """
    if 'vrp' in args.problem:
        from rm.data_generators.vrp import DataGenerator
        return DataGenerator

    elif 'cm' in args.problem:
        from rm.data_generators.cm import DataGenerator
        return DataGenerator

    else:
        raise Exception(f"Not a valid problem: {args.problem}")


def factory_linear_model(args):
    """ Gets factory LinearModel. """
    if 'vrp' in args.problem:
        from rm.sl_models.vrp import LinearModel
        return LinearModel

    elif 'cm' in args.problem:
        from rm.sl_models.cm import LinearModel
        return LinearModel

    else:
        raise Exception(f"Not a valid problem: {args.problem}")


def factory_input_invariant_model(args):
    """ Gets factory InputInvariantModel. """
    if 'vrp' in args.problem:
        from rm.sl_models.vrp import GraphNetwork
        return GraphNetwork

    elif 'cm' in args.problem:
        from rm.sl_models.cm import SetNetwork
        return SetNetwork

    else:
        raise Exception(f"Not a valid problem: {args.problem}")


def factory_policy_evaluator(args):
    """ Gets factory InputInvariantModel. """
    if 'vrp' in args.problem:
        from rm.envs.vrp.policy_evaluator import PolicyEvaluator
        return PolicyEvaluator

    elif 'cm' in args.problem:
        from rm.envs.cm.policy_evaluator import PolicyEvaluator
        return PolicyEvaluator

    else:
        raise Exception(f"Not a valid problem: {args.problem}")