# Medical Insurance Cost Prediction

This project predicts individual medical insurance charges from age, sex, BMI, number of children, smoker status, and region. The model is a linear regression trained with gradient descent, implemented by hand in NumPy instead of calling a library's fitting function, so the optimization step is visible in the code.

![Cost over training iterations](docs/cost_curve.png)

## Results

On a held-out test set of 268 records, the model explains 75% of the variance in charges (R² = 0.75) and its predictions are off by about $4,058 on average.

The data is split 80/20 into 1,070 training and 268 test records, using a fixed random seed.

## How it works

The model predicts a cost as a weighted sum of the six features plus a bias term:

$$\hat{y} = \mathbf{w}\cdot\mathbf{x} + b$$

Training minimizes the mean squared error, divided by two to simplify the gradient:

$$J(\mathbf{w}, b) = \frac{1}{2m}\sum_{i=1}^{m}\left(\hat{y}^{(i)} - y^{(i)}\right)^2$$

Each step moves the weights against the gradient:

$$w_j \leftarrow w_j - \alpha \frac{1}{m}\sum_{i=1}^{m}\left(\hat{y}^{(i)} - y^{(i)}\right)x_j^{(i)}$$

$$b \leftarrow b - \alpha \frac{1}{m}\sum_{i=1}^{m}\left(\hat{y}^{(i)} - y^{(i)}\right)$$

The learning rate is 0.01, and training runs for 1,000 iterations starting from zero weights. The cost curve above shows it leveling off well before the end.

Before training, sex, smoker status, and region are converted to numbers (sex: female = 1, male = 0; smoker: yes = 1, no = 0; region: northwest = 1, northeast = 2, southeast = 3, southwest = 4). Each feature is then standardized using the training set's mean and standard deviation. The test set is scaled with those same values, so no information from the test set leaks into training.

## What the model learned

The weights are in standardized units, so they can be compared directly:

| Feature | Weight |
|---|---|
| Smoker | 9,559 |
| Age | 3,649 |
| BMI | 1,933 |
| Children | 581 |
| Sex (female = 1) | 25 |
| Region | −312 |

Smoking has by far the largest effect, followed by age and BMI. Sex barely changes the prediction, and region has a small negative weight. The bias term is 13,314.

## Limitations

Region is encoded as 1 to 4, which implies an ordering the categories don't have. One-hot encoding would be more appropriate. A linear model also can't capture curved relationships or interactions, and medical costs likely depend on combinations such as smoking and BMI together. With only 1,338 rows, the error estimate is noisy, and the 0.75 R² could shift with a different split.

## Possible extensions

- One-hot encode region
- Add interaction and polynomial terms
- Compare against scikit-learn's `LinearRegression`
- Use k-fold cross-validation instead of a single split

## Project structure

```
medical-insurance-linear-regression/
├── docs/
│   └── cost_curve.png
├── linear_regression.ipynb
└── README.md
```
