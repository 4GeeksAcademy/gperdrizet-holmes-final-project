'''Collection of helper functions for notebooks.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import FunctionTransformer


def _sin(x, period):
    return np.sin(x / period * 2 * np.pi)

def _cos(x, period):
    return np.cos(x / period * 2 * np.pi)

def sin_transform(period):
    return FunctionTransformer(_sin, kw_args={'period': period})

def cos_transform(period):
    return FunctionTransformer(_cos, kw_args={'period': period})


def train_evaluate_model(
        model: callable,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        label: str,
        plot_title: str,
        label_transformer: callable=None,
        log_axes: bool=True
) -> callable:
    '''Takes model instance, train and test dfs, feature name and plot title string, trains and
    evaluates model with R2, RMSE. Plots predicted vs actual, fit residuals and normal QQ plot.
    returns fitted model instance'''

    _ = model.fit(train_df.drop(label, axis=1), train_df[label])

    training_predictions = model.predict(train_df.drop(label, axis=1))
    training_labels = train_df[label]

    if label_transformer is not None:
        training_predictions = label_transformer.inverse_transform(training_predictions.reshape(-1, 1))
        training_labels = label_transformer.inverse_transform(training_labels.values.reshape(-1, 1))

    predictions = model.predict(test_df.drop(label, axis=1))
    labels = test_df[label]

    if label_transformer is not None:
        predictions = label_transformer.inverse_transform(predictions.reshape(-1, 1))
        labels = label_transformer.inverse_transform(labels.values.reshape(-1, 1))
    
    residuals = labels - predictions
    standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
    residual_quantiles = np.percentile(standardized_residuals, [0, 25, 50, 75, 100])
    normal_quantiles = np.percentile(np.random.normal(size=1000), [0, 25, 50, 75, 100])

    training_rsq = r2_score(training_labels, training_predictions)
    training_rmse = root_mean_squared_error(training_labels, training_predictions)

    print(f'Training R\u00b2 = {training_rsq:.3f}')
    print(f'Training RMSE = {training_rmse:.3f}\n')

    rsq = r2_score(labels, predictions)
    rmse = root_mean_squared_error(labels, predictions)

    print(f'Testing R\u00b2 = {rsq:.3f}')
    print(f'Testing RMSE = {rmse:.3f}\n')

    fig, axs = plt.subplots(1, 3, figsize=(10, 3.5))

    fig.suptitle(f'{plot_title}\nR\u00b2 = {rsq:.3f}, RMSE = {rmse:.3f}')

    axs[0].set_title('True vs predicted')
    axs[0].scatter(predictions, labels, color='black', s=0.5, alpha=0.5)
    axs[0].axline((0, 0), slope=1, color='red', linestyle='--', label='Ideal fit')
    axs[0].set_ylabel(f'True {label}')
    axs[0].set_xlabel(f'Predicted {label}')

    if log_axes is True:
        axs[0].set_yscale('log')
        axs[0].set_xscale('log')

    axs[0].legend(loc='best')

    axs[1].set_title('Fit residuals')
    axs[1].scatter(predictions, residuals, color='black', s=0.5, alpha=0.5)
    axs[1].axhline(0, color='red', linestyle='--', label='Ideal fit')
    axs[1].set_xlabel(f'Predicted {label}')
    axs[1].set_ylabel(f'True - predicted {label}')


    axs[2].set_title('Residual normal probability')
    axs[2].scatter(normal_quantiles, residual_quantiles, color='black')
    axs[2].axline((normal_quantiles[0], residual_quantiles[0]), slope=1, color='red', linestyle='--', label='Ideal fit')
    axs[2].set_xlabel('Normal quantiles')
    axs[2].set_ylabel('Residual quantiles')

    fig.tight_layout()

    return model


def plot_cross_validation(search_results: GridSearchCV, plot_training: bool=False) -> None:
    '''Takes result object from scikit-learn's GridSearchCV() or RandomSearchCV(),
    draws plot of hyperparameter set validation score rank vs validation scores.'''

    results = pd.DataFrame(search_results.cv_results_)
    results = results[results['mean_test_score'] > 0]
    sorted_results = results.sort_values('rank_test_score')

    plt.title('Hyperparameter optimization')
    plt.xlabel('Hyperparameter set rank')
    plt.ylabel('Validation R\u00b2')
    plt.gca().invert_xaxis()

    plt.fill_between(
        sorted_results['rank_test_score'],
        sorted_results['mean_test_score'] + sorted_results['std_test_score'],
        sorted_results['mean_test_score'] - sorted_results['std_test_score'],
        alpha=0.5
    )

    plt.plot(
        sorted_results['rank_test_score'],
        sorted_results['mean_test_score'],
        label='Validation'
    )

    if plot_training:

        plt.fill_between(
            sorted_results['rank_test_score'],
            sorted_results['mean_train_score'] + sorted_results['std_train_score'],
            sorted_results['mean_train_score'] - sorted_results['std_train_score'],
            alpha=0.5
        )

        plt.plot(
            sorted_results['rank_test_score'],
            sorted_results['mean_train_score'],
            label='Training'
        )

        plt.legend(loc='best', fontsize='small')

    plt.show()
