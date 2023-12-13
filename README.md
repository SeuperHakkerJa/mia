# Membership Inference Attack (MIA)

This repository includes an implementation of a Membership Inference Attack (MIA) algorithm using shadow models. 
Additionally, it provides two protective mechanisms against MIA attacks, namely 'Labeler' and 'Ensemble.'


## Usage Example -- Attack

```python
import pandas

from mia import attack, protection

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# TRAIN MODEL
data = pandas.read_csv("../datasets//adult.csv")
data = data.apply(LabelEncoder().fit_transform)

X = data.drop(["class"], axis=1).to_numpy()
y = data["class"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

target = RandomForestClassifier().fit(X_train, y_train)
target.score(X_test, y_test)

# ATTACK
sample = data.sample(SAMPLE)

X = sample.drop(["class"], axis=1).to_numpy()
y = sample["class"].to_numpy()

factory = lambda: RandomForestClassifier()

sample = data.sample(200)

X = target.predict_proba(sample.drop(["class"], axis=1).to_numpy())
y = sample["class"].to_numpy()

attack = attack.MIA(factory=factory, categories=2, shadows=SHADOWS).fit(X, y)
attack.score(X, y)
```

## Usage Example -- Protections


```python
X = data.drop(["class"], axis=1).to_numpy()
y = data["class"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

target = protection.Labeler(factory).fit(X_train, y_train)
target.score(X_test, y_test)
```

```python
X = data.drop(["class"], axis=1).to_numpy()
y = data["class"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

target = protection.Ensemble(factory, n_estimators=10).fit(X_train, y_train)
target.score(X_test, y_test)
```

## Installation

This project is managed using Poetry. To install the project, you can use the following commands:

```bash
pip install poetry
poetry install
```

## Citation for the Original Authors

```bibtex
@INPROCEEDINGS{7958568,
  author={Shokri, Reza and Stronati, Marco and Song, Congzheng and Shmatikov, Vitaly},
  booktitle={2017 IEEE Symposium on Security and Privacy (SP)}, 
  title={Membership Inference Attacks Against Machine Learning Models}, 
  year={2017},
  volume={},
  number={},
  pages={3-18},
  doi={10.1109/SP.2017.41}
}
```
