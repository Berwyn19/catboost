# Folder Structure
- **datasets/**: Contains `train`, `validation`, and `test` datasets in CSV format.
- **config.yaml**: Stores model hyperparameters and specifies categorical features.
- **train.py**: Handles both training and testing.

# Training
python train.py \
  --train: path to the training dataset \
  --eval: path to the validation dataset \
  --save: path where the trained model will be saved \
  --config: path to the yaml file \
  --mode: "train" for training

# Evaluation
python train.py \
  --test: path to the test dataset \
  --save: path to the saved model

# Requirements
pip install -r requirements.txt

