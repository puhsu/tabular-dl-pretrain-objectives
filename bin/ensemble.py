# %%
import argparse

import numpy as np
import zero
from scipy.special import expit, softmax

import lib


# %%
parser = argparse.ArgumentParser("ensemble")
parser.add_argument('evaluation_dir', metavar='EVALUATION_DIR')
args = parser.parse_args()

experiment_dir = lib.get_path(args.evaluation_dir)

single_outputs = [
    [experiment_dir / str(seed) for seed in seeds]
    for seeds in [range(0,5), range(5,10), range(10,15)]
]

assert all((experiment_dir/f'{s}/DONE').exists() for s in range(15))

# %%
zero.improve_reproducibility(0)
parts = ['train', 'val', 'test']
first_report = lib.load_report(single_outputs[0][0])

D = lib.build_dataset(
    first_report['config']['data']['path'],
    lib.Transformations(normalization=first_report['config']['data']['T']['normalization']),
    cache=True,
)

Y, y_info = lib.build_target(
    D.y, first_report['config']['data']['T']['y_policy'], D.task_type
)

for i, seeds in enumerate(single_outputs):
    output = experiment_dir.with_name(
        experiment_dir.name.replace('evaluation', f'ensemble_5')
    )/str(i)
    output.mkdir(exist_ok=True, parents=True)

    report = {
        'program': 'bin/ensemble.py',
        'single_program': first_report['program'],
        'config': {
            'seeds': [s.name for s in seeds],
        },
        'prediction_type': None if D.is_regression else 'probs',
    }

    single_predictions = [lib.load_predictions(x) for x in seeds]

    predictions = {}
    for part in parts:
        stacked_predictions = np.stack([x[part] for x in single_predictions])  # type: ignore[code]
        if D.is_binclass:
            if not report['single_program'].find('catboost') != -1:
                stacked_predictions = expit(stacked_predictions)
        elif D.is_multiclass:
            if not (report['single_program'].find('catboost') != -1 or report['single_program'].find('xgboost') != -1):
                stacked_predictions = softmax(stacked_predictions, -1)
        else:
            assert D.is_regression

        predictions[part] = stacked_predictions.mean(0)

    report['metrics'] = D.calculate_metrics(predictions, report['prediction_type'])
    lib.finish(output, report)
