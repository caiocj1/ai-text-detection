import argparse
import os
import numpy as np
import pandas as pd
import yaml

from models.transformer_model import TransformerModel
from data_module import TextDataModule

import torch.cuda

from pytorch_lightning import Trainer

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', '-w', required=True)

    args = parser.parse_args()

    # Read config file
    config_path = os.path.join(os.getcwd(), 'config.yaml')
    with open(config_path) as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    model_params = params['ModelParams']

    data_module = TextDataModule(
        batch_size=8,
        num_workers=8
    )

    # Model selection
    model = TransformerModel(vocab_size=len(data_module.vocab),
                             d_model=model_params['d_model'],
                             nhead=model_params['nhead'],
                             dim_feedforward=model_params['dim_feedforward'],
                             num_layers=model_params['num_layers'],
                             dropout=model_params['dropout'])

    trainer = Trainer(accelerator='auto',
                      devices=1 if torch.cuda.is_available() else None)

    ckpt_path = os.path.join(args.weights_path)

    data_module.setup(stage='predict')

    test_results = trainer.predict(model, data_module, ckpt_path=ckpt_path, return_predictions=True)

    results = torch.argmax(torch.cat(test_results), dim=-1).numpy()

    test_ids = []
    for sample in data_module.X_test:
        test_ids.append(sample['id'])
    test_ids = np.array(test_ids)

    submission_df = pd.DataFrame(data={'id': test_ids, 'label': results})
    submission_df.to_csv('data/submission.csv', index=False)

    print('Saved csv')