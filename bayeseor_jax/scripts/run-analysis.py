"""CLI driver for BayesEoR-JAX setup and BlackJAX nested sampling."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from pprint import pformat

def main() -> None:
    """Parse config/CLI args, run setup, and optionally run nested sampling."""
    package_parent = Path(__file__).resolve().parents[2]
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))

    from bayeseor_jax.params import BayesEoRParser
    from bayeseor_jax.run import run
    from bayeseor_jax.setup import run_setup
    from bayeseor_jax.utils import write_log_files

    parser = BayesEoRParser()
    args = parser.parse_args()
    logging.basicConfig(
        level=(logging.WARNING if args.quiet else logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    if not args.quiet:
        logger.info("Parameters")
        if args.config:
            cfg_val = args.config[0] if isinstance(args.config, list) else args.config
            logger.info("Config file: %s", Path(cfg_val).absolute().as_posix())
        logger.info("%s", pformat(args.__dict__))

    args_dict = dict(vars(args))
    setup_result = run_setup(**args_dict)

    if not args.run:
        if not args.quiet:
            logger.info("Setup complete. Skipping nested sampling (`--run` not set).")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_log_files(parser, args, out_dir=out_dir)

    ns_out_dir = out_dir / "nested_sampling"
    summary = run(
        setup_result=setup_result,
        out_dir=ns_out_dir,
        random_key=args.random_key,
        n_live_scale_factor=args.n_live_scale_factor,
        num_inner_scale_factor=args.num_inner_scale_factor,
        delete_scale_factor=args.delete_scale_factor,
        stop_threshold=args.stop_threshold,
    )

    logger.info("Run summary:\n%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
