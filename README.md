# Biomechanial features analysis

## Abstract

Sagittal pelvic orientation features analysis is a data analysis in the field of medicine or more precisely orthopedics. Sagittal balance means a hramonious shape of the spine and the correct sagittal orientation of the pelvis is crucial in ensuring it. The sagittal orientation of the pelvis is determined on measuring biomechanical (geometric) parameters, which are then used to identify abnormalities that can cause the disease, such as disc herniation or spondylolisthesis.

Determining pelvic orientation abnormalities is commonly still manual task and, as a result, is a relatively subjective decision. With modern machine learning technics, patients can be classified and thus an easier and more accurate decision can be made. Also, the sagittal pelvic orientation features analysis discovers knowledge from the data of the patients with such diseases, such as which of the parameters is the most important for classification and how similar different individual groups of patients are.

## Data analysis

### Description of the data

|                          |   count |     mean |     std |       min |       25% |      50% |      75% |      max |
|:-------------------------|--------:|---------:|--------:|----------:|----------:|---------:|---------:|---------:|
| pelvic_incidence         |     310 |  60.4967 | 17.2365 |  26.1479  |  46.4303  |  58.691  |  72.8777 | 129.834  |
| pelvic_tilt              |     310 |  17.5428 | 10.0083 |  -6.55495 |  10.6671  |  16.3577 |  22.1204 |  49.4319 |
| lumbar_lordosis_angle    |     310 |  51.9309 | 18.5541 |  14       |  37       |  49.5624 |  63      | 125.742  |
| sacral_slope             |     310 |  42.9538 | 13.4231 |  13.3669  |  33.3471  |  42.4049 |  52.6959 | 121.43   |
| pelvic_radius            |     310 | 117.921  | 13.3174 |  70.0826  | 110.709   | 118.268  | 125.468  | 163.071  |
| degree_spondylolisthesis |     310 |  26.2967 | 37.559  | -11.0582  |   1.60373 |  11.7679 |  41.2874 | 418.543  |

### Class count

![system schema](img/class_count.png)

### Parameters distribution

![system schema](img/parameters_distribution.png)

### Pairwise relationship

![system schema](img/pairwise_relationship.png)

### Data correlation

#### Feature importance

![system schema](img/feature_importance.png)

#### Heat map

![system schema](img/heatmap.png)

### Hierarchical clustering

![system schema](img/dendrogram.png)

### PCA

#### Two components PCA

![system schema](img/PCA_two_components.png)

Explained variance per component:

|   PC1       |        PC2  |      sum  |
|-------------|-------------|-----------|
|  0.70963571 |  0.13759529 | 0.847231  |

#### Three components PCA

![system schema](img/PCA_three_components.png)

Explained variance per component:

|   PC1       |        PC2  |      PC3   | sum        |
|-------------|-------------|------------|------------|
|  0.70963571 |  0.13759529 | 0.07521804 |  0.92244904|

#### Explained variance ratio

![system schema](img/PCA_explained_variance_ratio.png)

### Prediction

#### Knn

![system schema](img/knn.png)

```
Optimal k for KNN calculated:  16

Cross validation mean accuracy:                 0.842741935483871
KNN prediction model accuracy (on test data):   0.8387096774193549
```

#### Logistic regression

```
Cross validation mean accuracy:                             0.842741935483871
Log regression prediction model accuracy (on test data):    0.8548387096774194
```

![system schema](img/ROC.png)

#### Decision tree

```
Cross validation mean accuracy:                            0.8024193548387097
Decision tree prediction model accuracy (on test data):    0.8064516129032258
```