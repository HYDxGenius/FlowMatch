# FlowMatch — slim algorithm registry.
# We expose only FixMatch (baseline) and FlowMatch (ours). Other algorithms
# from the original USB codebase have been removed for this release.

from semilearn.core.utils import ALGORITHMS
name2alg = ALGORITHMS

# Trigger registration via side-effect imports.
from semilearn.algorithms import fixmatch        # noqa: F401  baseline
from semilearn.algorithms import flowmatch       # noqa: F401  ours


def get_algorithm(args, net_builder, tb_log, logger):
    if args.algorithm in ALGORITHMS:
        alg = ALGORITHMS[args.algorithm](
            args=args,
            net_builder=net_builder,
            tb_log=tb_log,
            logger=logger,
        )
        return alg
    raise KeyError(f"Unknown algorithm: {args.algorithm}")
