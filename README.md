# Folder Structure

- **datasets/**: Contains `train.csv`, `validation.csv`, and `test.csv` datasets in CSV format.
- **config/config.yaml**: Stores model hyperparameters and specifies categorical features.
- **train.py**: Handles both training and testing.

---

# Training

To train the model, run:

```bash
python train.py \
  --train path/to/train.csv \
  --eval path/to/validation.csv \
  --save path/to/save_model.pkl \
  --config path/to/config.yaml \
  --mode "train"
```

```text
Arguments:
--train     Path to the training dataset
--eval      Path to the validation dataset
--save      Path where the trained model will be saved (in .pkl format)
--config    Path to the YAML config file
--mode      Must be set to "train" to trigger training
```

---

# Evaluation

To evaluate or test the model, run:

```bash
python train.py \
  --test path/to/test.csv \
  --save path/to/save_model.pkl
  --mode "eval"
```

```text
Arguments:
--test      Path to the test dataset
--save      Path to the saved model file from training (in .pkl format)
--mode      Must be set to "eval" to trigger evaluation
```

---

# Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```


