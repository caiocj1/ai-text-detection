# AI Text Detection

Code made for Kaggle challenge: https://www.kaggle.com/competitions/inf582-2023/overview

Part of École Polytechnique's (France) course INF582: Introduction to Text Mining and NLP.

All hyperparameters are available and can be changed in the ``config.yaml`` file. Current hyperparameters are the ones
that gave the best result in the leaderboard as explained in the report.

## Environment creation, tracking training

To create the environment, run ``conda env create -f environment.yaml``.

Set data set location with ``conda env config vars set DATASET_PATH=<path_to_csv_files>``.

To track training, ``tensorboard --logdir lightning_logs --bind_all``.

## Launch training

1. Run ``python generate_w2v.py`` to generate word2vec embeddings.
2. ``python run_training.py -v <version_name>``.
3. If you wish to run a new training with pre-loaded weights, add the option ``-w <path_to_ckpt>``.
4. To generate submission with a trained model, ``python run_prediction.py -w <path_to_ckpt_folder>``.

