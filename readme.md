# RoadCongestionPrediction

Course project, predict road congestion rate on a tiny dataset (total 122 samples). I utilized SMOTE method to up-sample the dataset. In addition, a gaussian noise was introduced to create more training data.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

This project is written in python and used tensorflow 1.12.

```
python			3.6
tensorflow		1.12
```

### Installing

Clone this repository.

```
git clone git@github.com:GlenGGG/RoadCongestionPrediction.git
cd RoadCongestionPrediction
```

### Usage

Simply run the code.

```
python congestionPrediction.py
```

Then you should be able to see some helpful information about the dataset and the training process. Cost of the training course will be printed every ten epochs. And finally, you will see a training curve displayed on a separate window.

