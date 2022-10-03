---
title: "Non-extensive introduction to Online Machine Learning"
draft: false
type: "page"
---

**Saulo Martiello Mastelini** (saulomastelini@gmail.com)

Contact:

- [Website](https://smastelini.github.io)
- [Github](https://github.com/smastelini)
- [Linkedin](https://www.linkedin.com/in/smastelini/)
- [ResearchGate](https://www.researchgate.net/profile/Saulo-Mastelini)

Copyright (c) 2022

---

**Disclaimer**

As the title implies, this material is not an extensive introduction to the topic. It is just my humble attempt to present a general overview of decades worth of research in an ever-expanding area.

Every process takes time. Therefore, a few minutes or hours are not enough to explore a whole research area. The idea is to find the balance between diving too deep into a topic and being too superficial. This hands-on talk is not formal, so feel free to interrupt me and ask questions anytime.

---

**If you want to explore further**

If you want to learn more about the topics discussed in this notebook, I suggest:

- [Knowledge discovery from data streams](http://www.liaad.up.pt/area/jgama/DataStreamsCRC.pdf) by the renowned researcher João Gama, 
- [MOA book](https://moa.cms.waikato.ac.nz/book-html/): an open-access book that discusses a lot of themes related to data streams
- [River documentation](https://riverml.xyz/): it has plenty of examples, tutorials, and theoretical resources. It is constantly updated and expanded.

If you have a specific question that is not covered in the documentation, you can always open a new _Discussion_ on Github. For sure somebody will help you! To do that, you need to head to the [River](https://github.com/online-ml/river) repository and find the discussion tab.

Contributions are always welcome. River is open source and kept by a community. Even though you might not have a technical background, it is always possible to help. Fixing and expanding the documentation is just an example of possible ways to get involved. If you find a bug, please let us know! 😁

---

**About River**

River is an open-source project focused on online machine learning and stream mining. It is the result of a merger between two preceding open source projects:

- creme
- scikit-multiflow

creme and scikit-multiflow had a lot of overlap and also different strengths and weaknesses. After a long time of planning and discussing core design aspects, the maintainers of both projects joined forces and created River.

Hence, River has the best of both worlds and it is the result of years of learned lessons in the preceding tools. River is focused on both researchers and practicioneers. A lot of people help River keep growing, but the core development team is spread between France, New Zealand, Vietnam, and Brazil.

---

## Outline

1. Online learning? Why?
2. Batch vs. Online
3. Building blocks: some examples
4. Why dictionaries?
5. How to evaluate an online machine learning model?
    - `progressive_val_score`
    - label delay
6. Concept drift
7. Examples of algorithms
    1. Classification
        1. Hoeffding Tree
        2. Adaptive Random Forest
    2. Regression
        1. **Hoeffding Tree**
        2. **AMRules**
    3. Clustering
        1. k-Means


```python
# Necessary packages

# !pip install numpy
# !pip install scikit-learn

# Latest released version
# !pip install river

# Development version
#!pip install git+https://github.com/online-ml/river --upgrade
```

# 1. Online Learning? Why?

Q: Why should somebody care about updating models online? What about just training them once and using them?
A: Well, that is indeed enough for most cases.

Nonetheless, imagine that:

- The amount of data instances is huge
- It is not possible to store everything
- The available computational power is limited
    - CPUs
    - Memory
    - Battery
- Data is non-stationary and/or evolves through time

Q: Is it possible to use traditional machine learning in these cases?
A: Yes!

One can still use traditional or batch machine learning if:

- Data is stationary, i.e., a sufficiently large sample is enough to achieve generalization

or

- The speed at which data is produced or collected is not too high
    - In these cases, the batch-incremental approach is a possible solution

## 1.1 Batch-incremental

A batch machine learning model is retrained in this strategy at regular intervals. Hence, we must define a training window by following one among the possible approaches:

<img src="time_windows.png">

**Fonte:** Adapted from:

> Carnein, M. and Trautmann, H., 2019. Optimizing data stream representation: An extensive survey on stream clustering algorithms. Business & Information Systems Engineering, 61(3), pp.277-297.

- *Landmarks* are the most common choice for batch-incremental applications. The window length is the central concern.
    - The current model may become outdated if the window is too large
    - The model may fail to capture the underlying patterns in the data if the window is too small.
    - Concept drift is a serious problem
        - Drifts do not typically occur at predefined and regular intervals
    
**Attention**: batch-incremental != mini-batch.

Artificial neural networks can be trained incrementally or progressively, usually relying on mini-batches of data.

Challenges such as "catastrophic forgetting" are one of the main concerns tackled in the **continual learning** research field.

## 1.2. It is worth noting

Data streams are not necessarily, time series.

Q: What is the difference between data streams and time series?
A: Data streams do not necessarily have explicit temporal dependencies like time series. For instance, sensor networks.
    - Varying transmission speeds
    - Sensor failure
    - Network expansion
    - And so on...
    Hence, the arrival order does not matter... much, but it does

# 2. Batch vs. Online

The River website has a nice [tutorial](https://riverml.xyz/latest/examples/batch-to-online/) on going from batch to online ML. But let's give a general overview of the differences.

A typical batch ML evaluation pipeline might look like this:


```python
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

data = load_wine()

X, y = data.data, data.target
kf = KFold(shuffle=True, random_state=8, n_splits=10)

accs = []

for train, test in kf.split(X):
    X_tr, X_ts = X[train], X[test]
    y_tr, y_ts = y[train], y[test]
    
    dt  = DecisionTreeClassifier(max_depth=5, random_state=93)
    dt.fit(X_tr, y_tr)
    
    accs.append(accuracy_score(y_ts, dt.predict(X_ts)))

print(f"Mean accuracy: {sum(accs) / len(accs)}")
```

    Mean accuracy: 0.9045751633986928



```python
len(X)
```




    178



The dataset is loaded in the memory and entirely available for inspection. The decision tree algorithm is allowed to perform multiple passes over the (training) data. Validation data is never used for training.

In the end, we might take the complete dataset (training + validation) to build a "final model", given that we have already found a good set of hyperparameters. Once trained, this model will be used to predict the types of wine samples.

Let's see what an online ML evaluation might look like:


```python
from river import metrics
from river import stream
from river import tree


acc = metrics.Accuracy()
ht = tree.HoeffdingTreeClassifier(max_depth=5, grace_period=20)

for x, y in stream.iter_sklearn_dataset(load_wine()):
    # The evaluation metric is evaluated before the model actually learns from the instance
    acc.update(y, ht.predict_one(x))
    # The model is updated one instance at a time
    ht.learn_one(x, y)

print(f"Accuracy: {acc.get()}")
```

    Accuracy: 0.9269662921348315



```python
x, y
```




    ({'alcohol': 14.13,
      'malic_acid': 4.1,
      'ash': 2.74,
      'alcalinity_of_ash': 24.5,
      'magnesium': 96.0,
      'total_phenols': 2.05,
      'flavanoids': 0.76,
      'nonflavanoid_phenols': 0.56,
      'proanthocyanins': 1.35,
      'color_intensity': 9.2,
      'hue': 0.61,
      'od280/od315_of_diluted_wines': 1.6,
      'proline': 560.0},
     2)



We process the input data sequentially. Data might be loaded on demand from the disk, a web server, or anywhere.
Data does not need to fit into the available memory.

Each instance is first used for testing and then to update the learning model. Everything works in an instance-by-instance regimen.

If the underlying process is guaranteed to be stationary, we could shuffle the data before passing it to the model.

**Note:** we cannot directly compare both the obtained accuracy values, as the evaluation strategies are not the same.

# 3. Building blocks: some examples


```python
import numpy as np
from river import stats
```

After first glancing at the differences, let's take things slowly and reflect on the building blocks necessary to perform Online Machine Learning.

Let's suppose we want to keep statistics for continually arriving data. For instance, we want to calculate the mean and variance.


Time to simulate:


```python
import random

rng = random.Random(42)
```


```python
%%time

values = []
stds_batch = []

for _ in range(50000):
    v = rng.gauss(5, 3)
    values.append(v)

    stds_batch.append(np.std(values, ddof=1) if len(values) > 1 else 0)
```

    CPU times: user 35 s, sys: 88.7 ms, total: 35.1 s
    Wall time: 35.1 s



```python
rng = random.Random(42)
```


```python
%%time

stds_incr = []
var = stats.Var(ddof=1)

for _ in range(50000):
    v = rng.gauss(5, 3)
    var.update(v)
    stds_incr.append(var.get() ** 0.5)
```

    CPU times: user 57.8 ms, sys: 1.14 ms, total: 58.9 ms
    Wall time: 58.4 ms


A lot faster! But does it work?


```python
s_errors = 0

for batch, incr in zip(stds_batch, stds_incr):
    s_errors += (batch - incr)

s_errors, s_errors / len(stds_batch)
```




    (-1.4842460593911255e-11, -2.968492118782251e-16)



I hope this is convincing! River's [stats](https://riverml.xyz/dev/api/overview/#stats) module has a lot of tools to calculate statistics 🧐

Many of these things are the building blocks of Online Machine Learning algorithms.

---

**Practical example: Variance using the Welford algorithm**

- We need some variables:
    - $n$: number of observations
    - $\overline{x}_n$: the sample mean, after $n$ observations
    - $M_{2, n}$: second-order statistic
- The variables are initialized as follows:
    - $\overline{x}_{0} \leftarrow 0$
    - $M_{2,0} \leftarrow 0$
- The variables are updated using the following expressions:
    - $\overline{x}_n = \overline{x}_{n-1} + \dfrac{x_n - \overline{x}_{n-1}}{n}$
    - $M_{2,n} = M_{2,n-1} + (x_n - \overline{x}_{n-1})(x_n - \overline{x}_n)$
- The sample variance is obtained using: $s_n^2 = \dfrac{M_{2,n}}{n-1}$, for every $n > 1$
- We also get a robust mean estimator for free! 🤓

---

# 4. Why dictionaries (or why using a sparse data representation)?

In River, we use dictionaries as the primary data type.

Dictionaries:

- Key x value: keys are unique
- Values accessed via keys instead of indices
- Sparse
- There is no explicit ordering
- Dynamic!
- Mixed data types

Examples:


```python
from datetime import datetime

x = {
    "potato": 3,
    "car": 2,
    "data": datetime.now(),
    "yes_or_no": "yes"
}

x
```




    {'potato': 3,
     'car': 2,
     'data': datetime.datetime(2022, 10, 3, 17, 39, 41, 659462),
     'yes_or_no': 'yes'}




```python
x["one extra"] = True
x
```




    {'potato': 3,
     'car': 2,
     'data': datetime.datetime(2022, 10, 3, 17, 39, 41, 659462),
     'yes_or_no': 'yes',
     'one extra': True}




```python
del x["data"]
x
```




    {'potato': 3, 'car': 2, 'yes_or_no': 'yes', 'one extra': True}



**Tip**: dictionaries are very similar to JSON.

Let's compare dictionaries with the traditional approach, based on arrays:


```python
data = load_wine()

X, y = data.data, data.target

X[0, :], data.feature_names
```




    (array([1.423e+01, 1.710e+00, 2.430e+00, 1.560e+01, 1.270e+02, 2.800e+00,
            3.060e+00, 2.800e-01, 2.290e+00, 5.640e+00, 1.040e+00, 3.920e+00,
            1.065e+03]),
     ['alcohol',
      'malic_acid',
      'ash',
      'alcalinity_of_ash',
      'magnesium',
      'total_phenols',
      'flavanoids',
      'nonflavanoid_phenols',
      'proanthocyanins',
      'color_intensity',
      'hue',
      'od280/od315_of_diluted_wines',
      'proline'])




```python
y[0], data.target_names
```




    (0, array(['class_0', 'class_1', 'class_2'], dtype='<U7'))



We are going to put sklearn to the test.


```python
X_tr, y_tr = X[:-2, :], y[:-2]
X_ts, y_ts = X[-2:, :], y[-2:]

X_tr.shape, X_ts.shape
```




    ((176, 13), (2, 13))




```python
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_tr, y_tr)

nb.predict(X_ts)
```




    array([2, 2])



What if one feature was missing?


```python
try:
    nb.predict(X_ts[:, 1:])
except ValueError as error:
    print(error)
```

    X has 12 features, but GaussianNB is expecting 13 features as input.


That type of situation is not uncommon in online scenarios. New sensors appear, some fail, and so on. So we must be able to deal with this kind of situation.

The majority of the models in River can deal with missing and emerging features! 🎉


```python
from river import naive_bayes

gnb = naive_bayes.GaussianNB()
dataset = stream.iter_sklearn_dataset(load_wine())

rng = random.Random(42)

# Probability of ignoring a feature
del_chance = 0.2

n_incomplete = 0
for i, (x, y) in enumerate(dataset):
    if i == 176:
        break
    
    x_copy = x.copy()
    aux = 0
    for xi in x:
        if rng.random() <= del_chance:
            del x_copy[xi]
            aux = 1
        
        # Update the counter of incomplete instances
        n_incomplete += aux
    
    gnb.learn_one(x_copy, y)
```


```python
x, y
```




    ({'alcohol': 13.17,
      'malic_acid': 2.59,
      'ash': 2.37,
      'alcalinity_of_ash': 20.0,
      'magnesium': 120.0,
      'total_phenols': 1.65,
      'flavanoids': 0.68,
      'nonflavanoid_phenols': 0.53,
      'proanthocyanins': 1.46,
      'color_intensity': 9.3,
      'hue': 0.6,
      'od280/od315_of_diluted_wines': 1.62,
      'proline': 840.0},
     2)




```python
gnb.predict_proba_one(x)
```




    {0: 2.2901730526820806e-23, 1: 4.523692607178262e-14, 2: 0.9999999999999538}



We are going to explicitly modify this last example:


```python
x, y = next(dataset)
list(x.keys())
```




    ['alcohol',
     'malic_acid',
     'ash',
     'alcalinity_of_ash',
     'magnesium',
     'total_phenols',
     'flavanoids',
     'nonflavanoid_phenols',
     'proanthocyanins',
     'color_intensity',
     'hue',
     'od280/od315_of_diluted_wines',
     'proline']



Firstly, we make a copy and delete some features:


```python
x_copy = x.copy()

del x_copy["malic_acid"]
del x_copy["hue"]
del x_copy["flavanoids"]

x_copy
```




    {'alcohol': 14.13,
     'ash': 2.74,
     'alcalinity_of_ash': 24.5,
     'magnesium': 96.0,
     'total_phenols': 2.05,
     'nonflavanoid_phenols': 0.56,
     'proanthocyanins': 1.35,
     'color_intensity': 9.2,
     'od280/od315_of_diluted_wines': 1.6,
     'proline': 560.0}



Will our model work?


```python
gnb.predict_proba_one(x_copy), y
```




    ({0: 7.394823717897268e-13, 1: 8.511456030879924e-13, 2: 0.9999999999984084},
     2)



What if new features appeared?


```python
x["1st extra"] = 7.89
x["2nd extra"] = 2

x
```




    {'alcohol': 14.13,
     'malic_acid': 4.1,
     'ash': 2.74,
     'alcalinity_of_ash': 24.5,
     'magnesium': 96.0,
     'total_phenols': 2.05,
     'flavanoids': 0.76,
     'nonflavanoid_phenols': 0.56,
     'proanthocyanins': 1.35,
     'color_intensity': 9.2,
     'hue': 0.61,
     'od280/od315_of_diluted_wines': 1.6,
     'proline': 560.0,
     '1st extra': 7.89,
     '2nd extra': 2}




```python
gnb.learn_one(x, y)

gnb.predict_one({"1st extra": 7.8, "2nd extra": 1.5})
```




    1




```python
np.unique(data.target, return_counts=True)
```




    (array([0, 1, 2]), array([59, 71, 48]))



Each model implements different strategies to deal with missing or emerging features.

In our example, "1" was the majority class, and so was the prediction of GaussianNB. That is the best it can do since there is not enough information about the new features. But these new features are already part of the model and will be updated with more observations.


```python
gnb.gaussians
```




    defaultdict(functools.partial(<class 'collections.defaultdict'>, <class 'river.proba.gaussian.Gaussian'>),
                {0: defaultdict(river.proba.gaussian.Gaussian,
                             {'alcohol': 𝒩(μ=13.751, σ=0.434),
                              'ash': 𝒩(μ=2.473, σ=0.234),
                              'alcalinity_of_ash': 𝒩(μ=16.896, σ=2.671),
                              'magnesium': 𝒩(μ=107.082, σ=10.720),
                              'total_phenols': 𝒩(μ=2.811, σ=0.286),
                              'flavanoids': 𝒩(μ=2.955, σ=0.367),
                              'proanthocyanins': 𝒩(μ=1.881, σ=0.399),
                              'hue': 𝒩(μ=1.077, σ=0.119),
                              'od280/od315_of_diluted_wines': 𝒩(μ=3.131, σ=0.348),
                              'malic_acid': 𝒩(μ=1.982, σ=0.661),
                              'nonflavanoid_phenols': 𝒩(μ=0.288, σ=0.071),
                              'color_intensity': 𝒩(μ=5.443, σ=1.342),
                              'proline': 𝒩(μ=1114.224, σ=228.521),
                              '1st extra': 𝒩(μ=0.000, σ=0.000),
                              '2nd extra': 𝒩(μ=0.000, σ=0.000)}),
                 1: defaultdict(river.proba.gaussian.Gaussian,
                             {'alcohol': 𝒩(μ=12.261, σ=0.549),
                              'ash': 𝒩(μ=2.228, σ=0.334),
                              'alcalinity_of_ash': 𝒩(μ=20.347, σ=3.468),
                              'magnesium': 𝒩(μ=94.476, σ=16.685),
                              'flavanoids': 𝒩(μ=2.040, σ=0.711),
                              'nonflavanoid_phenols': 𝒩(μ=0.371, σ=0.130),
                              'color_intensity': 𝒩(μ=3.128, σ=0.905),
                              'hue': 𝒩(μ=1.065, σ=0.184),
                              'od280/od315_of_diluted_wines': 𝒩(μ=2.751, σ=0.508),
                              'malic_acid': 𝒩(μ=1.995, σ=1.083),
                              'total_phenols': 𝒩(μ=2.288, σ=0.548),
                              'proanthocyanins': 𝒩(μ=1.687, σ=0.617),
                              'proline': 𝒩(μ=523.582, σ=169.681),
                              '1st extra': 𝒩(μ=0.000, σ=0.000),
                              '2nd extra': 𝒩(μ=0.000, σ=0.000)}),
                 2: defaultdict(river.proba.gaussian.Gaussian,
                             {'alcohol': 𝒩(μ=13.178, σ=0.544),
                              'malic_acid': 𝒩(μ=3.375, σ=1.205),
                              'alcalinity_of_ash': 𝒩(μ=21.650, σ=2.332),
                              'magnesium': 𝒩(μ=98.128, σ=10.416),
                              'nonflavanoid_phenols': 𝒩(μ=0.439, σ=0.137),
                              'proanthocyanins': 𝒩(μ=1.168, σ=0.433),
                              'color_intensity': 𝒩(μ=7.081, σ=2.336),
                              'hue': 𝒩(μ=0.687, σ=0.105),
                              'od280/od315_of_diluted_wines': 𝒩(μ=1.692, σ=0.272),
                              'proline': 𝒩(μ=620.488, σ=110.735),
                              'ash': 𝒩(μ=2.451, σ=0.183),
                              'total_phenols': 𝒩(μ=1.723, σ=0.348),
                              'flavanoids': 𝒩(μ=0.763, σ=0.276),
                              '1st extra': 𝒩(μ=7.890, σ=0.000),
                              '2nd extra': 𝒩(μ=2.000, σ=0.000)})})



# 5. How to evaluate models?


In every example presented so far, when a new instance arrives, we first make a prediction and then use the new datum to update the model.
No cross-validation, leave-one-out, and so on.

This evaluation strategy is close to a real-world scenario: usually, we first get the inputs without labels, and predictions must be made. After some time, class labels arrive.
In our examples, the label is "revealed" after the model makes a prediction. A delay exists between predicting and getting the label in an even more realistic evaluation scenario. Sometimes, the label never arrives for some instances.

We call this type of evaluation strategy _progressive validation_ or _prequential_ evaluation.

I suggest checking this [blog post from Max Halford](https://maxhalford.github.io/blog/online-learning-evaluation/), for more details on that matter.

In River, we have a utility function `progressive_val_score` in the `evaluate` module that handles all the situations mentioned above.


```python
from river import evaluate
from river import metrics
from river.datasets import synth


def label_delay(x, y):
    return rng.randint(0, 100)


rng = random.Random(8)
dataset = synth.RandomRBF(seed_sample=7, seed_model=9)
model = tree.HoeffdingTreeClassifier()

# We can combine metrics using pipeline operators
metric = metrics.Accuracy() + metrics.MicroF1() + metrics.BalancedAccuracy()

evaluate.progressive_val_score(
    dataset=dataset.take(50000),
    model=model,
    metric=metric,
    print_every=5000,
    show_memory=True,
    show_time=True,
    delay=label_delay
)
```

    [5,000] Accuracy: 69.65%, MicroF1: 69.65%, BalancedAccuracy: 69.12% – 00:00:00 – 84.73 KB
    [10,000] Accuracy: 72.09%, MicroF1: 72.09%, BalancedAccuracy: 71.42% – 00:00:01 – 90.98 KB
    [15,000] Accuracy: 73.55%, MicroF1: 73.55%, BalancedAccuracy: 73.12% – 00:00:01 – 156.3 KB
    [20,000] Accuracy: 75.62%, MicroF1: 75.62%, BalancedAccuracy: 75.24% – 00:00:02 – 199.85 KB
    [25,000] Accuracy: 78.32%, MicroF1: 78.32%, BalancedAccuracy: 78.01% – 00:00:02 – 243.42 KB
    [30,000] Accuracy: 80.24%, MicroF1: 80.24%, BalancedAccuracy: 80.01% – 00:00:03 – 308.74 KB
    [35,000] Accuracy: 81.68%, MicroF1: 81.68%, BalancedAccuracy: 81.50% – 00:00:03 – 374.06 KB
    [40,000] Accuracy: 82.90%, MicroF1: 82.90%, BalancedAccuracy: 82.74% – 00:00:04 – 439.07 KB
    [45,000] Accuracy: 83.83%, MicroF1: 83.83%, BalancedAccuracy: 83.68% – 00:00:05 – 460.84 KB
    [50,000] Accuracy: 84.60%, MicroF1: 84.60%, BalancedAccuracy: 84.45% – 00:00:05 – 498.76 KB





    Accuracy: 84.60%, MicroF1: 84.60%, BalancedAccuracy: 84.45%




```python
model.draw()
```




    
![svg](output_50_0.svg)
    



# 6. Concept drift

One of the main concerns in online machine learning is the fact that data distribution may not be stationary. What does that mean?

Let's first think about an example of stationary distribution:

> Big tech company X released a new neural network for the Y problem with 3 zillion parameters, trained for 6 months using enough energy to power up multiple cities. The training dataset had Z terabytes...

Well, the data does not change. Linguistic rules (in NLP) or visual semantics don't usually vary or evolve. Everything is static under the same data collection policy.

A dog will always be a dog. A word has a limited set of synonyms, and so on. The rule of the game does not change. But even in these scenarios, there are exceptions. What if the rules changed?

These changes or concept drift may occur in real-world problems. For example:

Consumer buying pattern (toilet paper, masks, and hand sanitizer at the beginning of Covid pandemics);
Renewable energy production: sunlight and wind are not predictable;
traffic and routes

An entire research field in online machine learning is devoted to creating concept drift detectors and learning algorithms capable of adapting to changes in the data distribution.

I am not an expert on this topic, but I will try to give you a simple example of how to apply a drift detector.

Let's suppose we have a classification problem and are monitoring our model's predictive performance. We denote by $0$ the cases where the model correctly classifies an instance and by $1$ the misclassifications.


```python
rng = random.Random(8)

for _ in range(10):
    print(rng.choices([0, 1], weights=[0.7, 0.3])[0])
```

    0
    1
    0
    1
    0
    0
    1
    0
    0
    0


We can feed these values to a drift detector:


```python
from river import drift

detector = drift.ADWIN(delta=0.01)

vals = rng.choices([0, 1], weights=[0.7, 0.3], k=500)
for i, v in enumerate(vals):
    detector.update(v)
    
    if detector.drift_detected:
        print(f"Drift detected: {i}")
```

What if the data distribution changes


```python
detector = drift.ADWIN(delta=0.05)

vals = rng.choices([0, 1], weights=[0.7, 0.3], k=500)
vals.extend(rng.choices([0, 1], weights=[0.2, 0.8], k=500))
for i, v in enumerate(vals):
    detector.update(v)
    
    if detector.drift_detected:
        print(f"Drift detected: {i}")
```

    Drift detected: 575


ADWIN is one of the most utilized drift detectors, but there many other algorithms. Non-supervised, semi-supervised, multivariate, and so on.
Usually, detectors are used as components of predictive models. Each models applies drift detectors in a different manner.

# 7. Algorithm examples

I will present some examples of classification, regression, and clustering algorithms for reference. The API access is always the same, so you can try your luck and check other examples in the documentation.


## 7.1. Classification

Algorithms projected for binary classification can be extended to the multiclass case by relying on the tools available in the `multiclass` module:

- `OneVsOneClassifier`
- `OneVsRestClassifier`
- `OutputCodeClassifier`

River also have basics tools to handle multi-output tasks. Any contributions are welcome!

### 7.1.1. Hoeffding Trees

One of the most popular families of online machine learning algorithms. They take this name because the statistical measure called Hoeffding bound is used to define when splits are performed. This heuristic ensures the decisions taken incrementally are similar to those performed by a batch decision tree algorithm.


There are three main variants of Hoeffding Trees:

- Hoeffding Tree: vanilla version
- Hoeffding Adaptive Tree: adds drift detectors to each decision node. If a drift is detected, a new subtree is trained in the background and eventually may replace the affected tree branch.
- Extremely Fast Decision Tree: quickly deploys splits but later revisits and improves its own decisions.

**Main hyperparameters:**

- `grace_period`: the interval between split attempts.
- `delta`: the split significance parameter. The split confidence `1 - delta`.
- `max_depth`: max height a tree might have.

I wrote a [tutorial](https://riverml.xyz/dev/user-guide/on-hoeffding-trees/) on Hoeffding Trees, where you can find more details about the algorithms.

**Example:**


```python
from river import tree


dataset = synth.RandomRBFDrift(
    seed_model=7, seed_sample=8, change_speed=0.0001, n_classes=3,
).take(15000)
model = tree.HoeffdingAdaptiveTreeClassifier(seed=42)
metric = metrics.Accuracy()

evaluate.progressive_val_score(dataset, model, metric, print_every=1000, show_memory=True, show_time=True)
```

    [1,000] Accuracy: 56.06% – 00:00:00 – 38.11 KB
    [2,000] Accuracy: 56.78% – 00:00:00 – 38.17 KB
    [3,000] Accuracy: 57.42% – 00:00:01 – 38.23 KB
    [4,000] Accuracy: 57.16% – 00:00:01 – 38.23 KB
    [5,000] Accuracy: 56.93% – 00:00:01 – 38.23 KB
    [6,000] Accuracy: 56.68% – 00:00:02 – 68.83 KB
    [7,000] Accuracy: 57.14% – 00:00:02 – 69.7 KB
    [8,000] Accuracy: 58.17% – 00:00:02 – 69.89 KB
    [9,000] Accuracy: 58.65% – 00:00:03 – 70.02 KB
    [10,000] Accuracy: 59.07% – 00:00:03 – 101.2 KB
    [11,000] Accuracy: 59.73% – 00:00:04 – 101.45 KB
    [12,000] Accuracy: 60.14% – 00:00:04 – 101.57 KB
    [13,000] Accuracy: 60.67% – 00:00:04 – 101.7 KB
    [14,000] Accuracy: 60.97% – 00:00:05 – 101.76 KB
    [15,000] Accuracy: 60.94% – 00:00:05 – 132.56 KB





    Accuracy: 60.94%



We can visualize the tree structure:


```python
model.draw()
```




    
![svg](output_63_0.svg)
    



We can also inspect how decisions are made:


```python
dataset = synth.RandomRBFDrift(
    seed_model=7, seed_sample=8, change_speed=0.0001, n_classes=3,
).take(15000)

x, y = next(dataset)

print(model.debug_one(x))
```

    4 ≤ 0.44069887341009073
    Class 2:
    	P(0) = 0.1
    	P(1) = 0.4
    	P(2) = 0.5
    


### 7.1.2. Adaptive Random Forest

Adaptive random forest (ARF) is an incremental version of Random Forests that combines the following ingredients:

- Randomized Hoeffding Trees as base learners
- Drifts detectors for each tree
    - New trees are trained in the background when drifts are detected
- Online bagging

ARFs have all the parameters of HTs and also some extra critical parameters:

- `warning_detector` and `drift_detector`
- `n_models`: the number of trees
- `max_features`: the maximum number of features considered during split attempts at each decision node

**Example:**


```python
from river import ensemble


dataset = synth.RandomRBFDrift(
    seed_model=7, seed_sample=8, change_speed=0.0001, n_classes=3,
).take(15000)

model = ensemble.AdaptiveRandomForestClassifier(seed=8)
metric = metrics.Accuracy()

evaluate.progressive_val_score(dataset, model, metric, print_every=1000, show_memory=True, show_time=True)
```

    [1,000] Accuracy: 65.67% – 00:00:01 – 1.04 MB
    [2,000] Accuracy: 71.39% – 00:00:02 – 1.07 MB
    [3,000] Accuracy: 73.06% – 00:00:03 – 2.27 MB
    [4,000] Accuracy: 74.59% – 00:00:04 – 2.68 MB
    [5,000] Accuracy: 75.98% – 00:00:06 – 3.96 MB
    [6,000] Accuracy: 77.31% – 00:00:07 – 5.13 MB
    [7,000] Accuracy: 77.85% – 00:00:08 – 3.8 MB
    [8,000] Accuracy: 78.22% – 00:00:10 – 5.09 MB
    [9,000] Accuracy: 78.25% – 00:00:11 – 6.41 MB
    [10,000] Accuracy: 78.52% – 00:00:12 – 7 MB
    [11,000] Accuracy: 78.36% – 00:00:14 – 8.38 MB
    [12,000] Accuracy: 78.49% – 00:00:15 – 7.64 MB
    [13,000] Accuracy: 78.58% – 00:00:17 – 8.97 MB
    [14,000] Accuracy: 78.53% – 00:00:18 – 10.32 MB
    [15,000] Accuracy: 78.41% – 00:00:20 – 10.76 MB





    Accuracy: 78.41%



## 7.2. Regression


I will use the same dataset for every regression example:


```python
def get_friedman():
    return synth.Friedman(seed=101).take(20000)
```


```python
x, y = next(get_friedman())
x, y
```




    ({0: 0.5811521325045647,
      1: 0.1947544955341367,
      2: 0.9652511070611112,
      3: 0.9239764016767943,
      4: 0.46713867819697397,
      5: 0.6634706445300605,
      6: 0.21452296973796803,
      7: 0.22169624952624067,
      8: 0.28852243338125616,
      9: 0.6924227459953175},
     20.0094162975429)



### 7.2.2. Hoeffding Tree

(I research this topic)

We have three main types of HTs for regression tasks:

- `HoeffdingTreeRegressor`: vanilla regressor.
- `HoeffdingAdaptiveTreeRegressor`: the regression counterpart of the adaptive classification tree.
- `iSOUPTreeRegressor`: Hoeffding Tree for multi-target regression tasks

Besides the parameters presented in the classification version, other important parameters are:

- `leaf_prediction`: the prediction strategy (regression or model tree)
- `leaf_model`: the regression model used in model trees' leaves
- `splitter`: the decision split algorithm


```python
from river import preprocessing

# We can combine multiple metrics in our report
metric = metrics.MAE() + metrics.RMSE() + metrics.R2()
model = preprocessing.StandardScaler() | tree.HoeffdingTreeRegressor()

evaluate.progressive_val_score(
    dataset=get_friedman(),
    model=model,
    metric=metric,
    show_memory=True,
    show_time=True,
    print_every=2000
)
```

    [2,000] MAE: 2.211476, RMSE: 2.922308, R2: 0.662991 – 00:00:00 – 975.25 KB
    [4,000] MAE: 2.06573, RMSE: 2.711252, R2: 0.706024 – 00:00:00 – 1.57 MB
    [6,000] MAE: 1.97555, RMSE: 2.570947, R2: 0.735164 – 00:00:00 – 2.19 MB
    [8,000] MAE: 1.911936, RMSE: 2.481903, R2: 0.753895 – 00:00:00 – 2.73 MB
    [10,000] MAE: 1.870939, RMSE: 2.424265, R2: 0.766415 – 00:00:01 – 3.27 MB
    [12,000] MAE: 1.834748, RMSE: 2.375678, R2: 0.774293 – 00:00:01 – 3.86 MB
    [14,000] MAE: 1.801622, RMSE: 2.329051, R2: 0.782718 – 00:00:01 – 4.36 MB
    [16,000] MAE: 1.773081, RMSE: 2.292944, R2: 0.790029 – 00:00:02 – 5.11 MB
    [18,000] MAE: 1.751902, RMSE: 2.264719, R2: 0.794645 – 00:00:02 – 6.19 MB
    [20,000] MAE: 1.728428, RMSE: 2.234173, R2: 0.800588 – 00:00:02 – 6.35 MB





    MAE: 1.728428, RMSE: 2.234173, R2: 0.800588



As usual, we can inspect how decisions are made:


```python
x, y = next(get_friedman())

print(model.debug_one(x))
```

    0. Input
    --------
    0: 0.58115 (float)
    1: 0.19475 (float)
    2: 0.96525 (float)
    3: 0.92398 (float)
    4: 0.46714 (float)
    5: 0.66347 (float)
    6: 0.21452 (float)
    7: 0.22170 (float)
    8: 0.28852 (float)
    9: 0.69242 (float)
    
    1. StandardScaler
    -----------------
    0: 0.28929 (float)
    1: -1.07485 (float)
    2: 1.58610 (float)
    3: 1.47168 (float)
    4: -0.11737 (float)
    5: 0.56379 (float)
    6: -0.99330 (float)
    7: -0.96557 (float)
    8: -0.73064 (float)
    9: 0.67069 (float)
    
    2. HoeffdingTreeRegressor
    -------------------------
    3 > -0.1
    1 ≤ -0.8
    3 > 0.6
    0 > -0.9
    4 > -0.9
    2 > -1.1
    1 > -1.5
    Mean: 15.269219 | Var: 8.424779
    
    
    Prediction: 18.29789



```python
model[-1].draw(3)
```




    
![svg](output_75_0.svg)
    



### 7.2.3. AMRules

Adaptive Model Rules.

(I also research this topic)

Creates decision rules by relying on the Hoeffding Bound. AMRules also has anomaly detection capabilities to "skip" anomalous training samples.

It has parameters similar to those of HTs:

- `n_min`: equivalent to `grace_period`
- `pred_type`: equivalent to `leaf_prediction`
- `pred_model`: equivalent to `leaf_model`
- `splitter`

Other important parameters:

- `m_min`: minimum number of instances to observe before detecting anomalies.
- `drift_detector`: the drift detection algorithm used by each rule.
- `anomaly_threshold`: threshold to decide whether or not an instance is anomalous (the smaller the score value, the more anomalous the instance is).
- `ordered_rule_set`: defines whether only the first rule is used for detection (when set to `True`) or all the rules are used (`False`).


```python
from river import rules

metric = metrics.MAE() + metrics.RMSE() + metrics.R2()
model = preprocessing.StandardScaler() | rules.AMRules(
    splitter=tree.splitter.TEBSTSplitter(digits=1),  #  <- this is part of my research
    drift_detector=drift.ADWIN(),
    ordered_rule_set=False,
    m_min=100,
    delta=0.01
)

evaluate.progressive_val_score(
    dataset=get_friedman(),
    model=model,
    metric=metric,
    show_memory=True,
    show_time=True,
    print_every=2000
)
```

    [2,000] MAE: 2.751126, RMSE: 3.585212, R2: 0.492754 – 00:00:00 – 557.26 KB
    [4,000] MAE: 2.594004, RMSE: 3.401369, R2: 0.537321 – 00:00:00 – 0.98 MB
    [6,000] MAE: 2.440782, RMSE: 3.200783, R2: 0.589509 – 00:00:00 – 1.06 MB
    [8,000] MAE: 2.35917, RMSE: 3.094535, R2: 0.617403 – 00:00:01 – 1.08 MB
    [10,000] MAE: 2.320284, RMSE: 3.045013, R2: 0.631478 – 00:00:01 – 1.29 MB
    [12,000] MAE: 2.280378, RMSE: 2.994163, R2: 0.641474 – 00:00:01 – 1.18 MB
    [14,000] MAE: 2.257398, RMSE: 2.963653, R2: 0.648179 – 00:00:02 – 1.51 MB
    [16,000] MAE: 2.267085, RMSE: 2.982856, R2: 0.644666 – 00:00:02 – 1.92 MB
    [18,000] MAE: 2.272122, RMSE: 2.98515, R2: 0.643212 – 00:00:03 – 2.08 MB
    [20,000] MAE: 2.26724, RMSE: 2.983389, R2: 0.644419 – 00:00:03 – 2.18 MB





    MAE: 2.26724, RMSE: 2.983389, R2: 0.644419



We can also inspect the model:


```python
x, y = next(get_friedman())

print(model.debug_one(x))
print(f"True label: {y}")
```

    0. Input
    --------
    0: 0.58115 (float)
    1: 0.19475 (float)
    2: 0.96525 (float)
    3: 0.92398 (float)
    4: 0.46714 (float)
    5: 0.66347 (float)
    6: 0.21452 (float)
    7: 0.22170 (float)
    8: 0.28852 (float)
    9: 0.69242 (float)
    
    1. StandardScaler
    -----------------
    0: 0.28929 (float)
    1: -1.07485 (float)
    2: 1.58610 (float)
    3: 1.47168 (float)
    4: -0.11737 (float)
    5: 0.56379 (float)
    6: -0.99330 (float)
    7: -0.96557 (float)
    8: -0.73064 (float)
    9: 0.67069 (float)
    
    2. AMRules
    ----------
    Default rule triggered:
    	Prediction (adaptive): 17.4743
    
    
    Prediction: 17.47431
    True label: 20.0094162975429



```python
x_scaled = model["StandardScaler"].transform_one(x)

model["AMRules"].anomaly_score(x_scaled)
```




    (-0.22130561042509, 0.0, 0)



## 7.3. Clustering

Incremental algorithms must adapt to changes in the data. For instance, new clusters might appear, some might disappear. I will show one example of algorithm:

### 7.3.1. k-Means

There are multiple incremental versions of k-Means out there. The version available in River adds a parameter called `halflife` which controls the the intensity of the incremental updates.



```python
from river import cluster

metric = metrics.Silhouette()
model = cluster.KMeans(seed=7)


for x, _ in get_friedman():
    metric.update(x, model.predict_one(x), model.centers)
    model.learn_one(x)

print(metric.get())
```

    0.24465881722583005



```python
metric = metrics.Silhouette()
model = cluster.KMeans(n_clusters=3, seed=7)


for x, _ in get_friedman():
    metric.update(x, model.predict_one(x), model.centers)
    model.learn_one(x)

print(metric.get())
```

    0.6612806222738018


And increase the `halflife` value.


```python
metric = metrics.Silhouette()
model = cluster.KMeans(n_clusters=3, seed=7, halflife=0.7)


for x, _ in get_friedman():
    metric.update(x, model.predict_one(x), model.centers)
    model.learn_one(x)

print(metric.get())
```

    0.7161109032425856


# Wrapping up

We can go much deeper into online machine learning solutions. There are multiple strategies to combine models, selecting the best model among a set thereof, and many other aspects. Online hyperparameter tuning is also an exciting research area.

I strongly suggest checking these additional resources to learn more about online machine learning:

**Tutorials:**

- [The art of using pipelines](https://riverml.xyz/latest/examples/the-art-of-using-pipelines/)
- [Working with imbalanced data](https://riverml.xyz/dev/examples/imbalanced-learning/)
- [Debbuging a pipeline](https://riverml.xyz/dev/examples/debugging-a-pipeline/)

**Resource hub:**

- [Awesome online machine learning](https://github.com/online-ml/awesome-online-machine-learning)


---

Thank you so much for having me!

Do you have any questions?
