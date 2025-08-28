# Notes

## 1. Sales regression

### 1.1. Without polynomial features

**Linear regression**:

- Training R² = 0.083
- Training RMSE = 433.070
- Testing R² = 0.088
- Testing RMSE = 418.105

**Gradient boosting**

- Training R² = 0.279
- Training RMSE = 383.992
- Testing R² = 0.174
- Testing RMSE = 397.868

### 1.2. With polynomial features

**Linear regression**

- Training R² = 0.131
- Training RMSE = 394.890
- Testing R² = 0.084
- Testing RMSE = 478.074

**Gradient boosting**

- Training R² = 0.401
- Training RMSE = 352.161
- Testing R² = 0.163
- Testing RMSE = 394.260

**Optimized gradient boosting (2k random search)**

- Training R² = 0.264
- Training RMSE = 390.421
- Testing R² = 0.197
- Testing RMSE = 386.112
- criterion: squared_error
- learning_rate: 0.040358382686161244
- loss: squared_error
- max_depth: 38
- max_features: 0.2493776797865007
- min_impurity_decrease: 1.7558076154643047e-09
- min_samples_leaf: 50
- min_samples_split: 528
- min_weight_fraction_leaf: 0.05315393132085977
- n_estimators: 181
- subsample: 0.753171342601556

### 1.3. Polynomial features + log(target)

**linear regression**

