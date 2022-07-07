import argparse
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import lib

parser = argparse.ArgumentParser()
parser.add_argument('tuning_output', metavar='DIR', type=Path)
parser.add_argument('n_seeds', type=int)
parser.add_argument('--force', action='store_true', help='override')
args = parser.parse_args()

assert args.tuning_output.name.endswith('_tuning')
assert (args.tuning_output / 'DONE').exists()
tuning_report = lib.load_report(args.tuning_output)
best_config = tuning_report['best']['config']

evaluation_dir: Path = args.tuning_output.with_name(
    args.tuning_output.name.replace('tuning', 'evaluation')
)
evaluation_dir.mkdir(exist_ok=True)
program = lib.get_temporary_copy(tuning_report['config']['program'])

for seed in range(args.n_seeds):
    config_path = evaluation_dir / f'{seed}.toml'
    config = deepcopy(best_config)
    config['seed'] = seed
    if tuning_report['config']['program'] == 'bin/archive/catboost_.py':
        config['catboost']['task_type'] = 'CPU'  # this is crucial for good results
        config['catboost']['thread_count'] = 32
    try:
        lib.dump_config(config, config_path)
        run_args = [lib.get_python(), program, config_path]

        if args.force or not (evaluation_dir/f"{seed}/DONE").exists():
            run_args.append('--force')
        subprocess.run(run_args, check=True)

    except Exception:
        config_path.unlink(True)
        shutil.rmtree(config_path.with_suffix(''), True)
        raise

lib.backup_output(evaluation_dir, force=True)
