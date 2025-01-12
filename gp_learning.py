import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style(
    style='darkgrid',
    rc={'axes.facecolor': '.9', 'grid.color': '.8'}
)
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')

import pymc as pm
import arviz as az

plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100

# az.rcParams["plot.backend"] = "bokeh"
# az.output_file("gp_learning.html")

np.random.seed(42)

# Generate seasonal variables.
def seasonal(t, amplitude, period):
    """Generate a sinusoidal curve."""
    return amplitude * np.sin((2*np.pi*t)/period)

def generate_data(n, sigma_n = 0.3):
    """Generate sample data.
    Two seasonal components, one linear trend and gaussian noise.
    """
    # Define "time" variable.
    t = np.arange(n)
    data_df = pd.DataFrame({'t' : t})
    # Add components:
    data_df['epsilon'] = np.random.normal(loc=0, scale=sigma_n, size=n)
    data_df['s1'] = data_df['t'].apply(lambda t: seasonal(t, amplitude=2, period=40))
    data_df['s2'] = data_df['t'].apply(lambda t: seasonal(t, amplitude=1, period=13.3))
    data_df['tr1'] = 0.01 * data_df['t']
    return data_df.eval('y = s1 + s2 + tr1 + epsilon')


# Number of samples.
n = 450
# Generate data.
data_df = generate_data(n=n)
# Plot.
fig, ax = plt.subplots()
sns.lineplot(x='t', y='y', data=data_df, color=sns_c[0], label='y', ax=ax)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set(title='Sample Data', xlabel='t', ylabel='')
plt.draw()

x = data_df['t'].values.reshape(n, 1)
y = data_df['y'].values.reshape(n, 1)

prop_train = 0.7
n_train = round(prop_train * n)

x_train = x[:n_train]
y_train = y[:n_train]

x_test = x[n_train:]
y_test = y[n_train:]

# Plot.
fig, ax = plt.subplots()
sns.lineplot(x=x_train.flatten(), y=y_train.flatten(), color=sns_c[0], label='y_train', ax=ax)
sns.lineplot(x=x_test.flatten(), y=y_test.flatten(), color=sns_c[1], label='y_test', ax=ax)
ax.axvline(x=x_train.flatten()[-1], color=sns_c[7], linestyle='--', label='train-test-split')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set(title='y train-test split ', xlabel='t', ylabel='')
plt.draw()

with pm.Model() as model:

    # First seasonal component.
    ls_1 = pm.Gamma(name='ls_1', alpha=2.0, beta=1.0)
    period_1 = pm.Gamma(name='period_1', alpha=80, beta=2)
    gp_1 = pm.gp.Marginal(cov_func=pm.gp.cov.Periodic(input_dim=1, period=period_1, ls=ls_1))
    # Second seasonal component.
    ls_2 = pm.Gamma(name='ls_2', alpha=2.0, beta=1.0)
    period_2 = pm.Gamma(name='period_2', alpha=30, beta=2)
    gp_2 = pm.gp.Marginal(cov_func=pm.gp.cov.Periodic(input_dim=1, period=period_2, ls=ls_2))
    # Linear trend.
    c_3 = pm.Normal(name='c_3', mu=1, sigma=2)
    gp_3 = pm.gp.Marginal(cov_func=pm.gp.cov.Linear(input_dim=1, c=c_3))
    # Define gaussian process.
    gp = gp_1 + gp_2 + gp_3
    # Noise.
    sigma = pm.HalfNormal(name='sigma', sigma=20)
    # Likelihood.
    y_pred = gp.marginal_likelihood('y_pred', X=x_train, y=y_train.flatten(), sigma=sigma)
    # Sample.
    print("Primary sampling")
    trace = pm.sample(draws=2000, chains=4, nuts_sampler_kwargs={"chain_method": "vectorized"}, tune=200, progressbar=True, nuts_sampler="blackjax")

# Plot parameters posterior distribution.
az_fig = az.plot_trace(trace)
plt.draw()

# Get model summary.
pm.summary(trace)

with model:
    x_train_conditional = gp.conditional('x_train_conditional', x_train)
    y_train_pred_samples = pm.sample_posterior_predictive(trace, var_names=["x_train_conditional"], predictions=True,)

    x_test_conditional = gp.conditional('x_test_conditional', x_test)
    y_test_pred_samples = pm.sample_posterior_predictive(trace, var_names=["x_test_conditional"], predictions=True,)

# Train
y_train_pred_samples_mean = y_train_pred_samples.predictions['x_train_conditional'].mean(("chain", "draw"))
y_train_pred_samples_std = y_train_pred_samples.predictions['x_train_conditional'].std()
y_train_pred_samples_mean_plus = y_train_pred_samples_mean + 2*y_train_pred_samples_std
y_train_pred_samples_mean_minus = y_train_pred_samples_mean - 2*y_train_pred_samples_std
# Test
y_test_pred_samples_mean = y_test_pred_samples.predictions['x_test_conditional'].mean(("chain", "draw"))
y_test_pred_samples_std = y_test_pred_samples.predictions['x_test_conditional'].std()
y_test_pred_samples_mean_plus = y_test_pred_samples_mean + 2*y_test_pred_samples_std
y_test_pred_samples_mean_minus = y_test_pred_samples_mean - 2*y_test_pred_samples_std


fig, ax = plt.subplots()
sns.lineplot(x=x_train.flatten(), y=y_train.flatten(), color=sns_c[0], label='y_train', ax=ax)
sns.lineplot(x=x_test.flatten(), y=y_test.flatten(), color=sns_c[1], label='y_test', ax=ax)
ax.fill_between(
    x=x_train.flatten(),
    y1=y_train_pred_samples_mean_minus,
    y2=y_train_pred_samples_mean_plus,
    color=sns_c[2],
    alpha=0.2,
    label='credible_interval (train)'
)
sns.lineplot(x=x_train.flatten(), y=y_train_pred_samples_mean, color=sns_c[2], label='y_pred_train', ax=ax)
ax.fill_between(
    x=x_test.flatten(),
    y1=y_test_pred_samples_mean_minus,
    y2=y_test_pred_samples_mean_plus,
    color=sns_c[3],
    alpha=0.2,
    label='credible_interval (test)'
)
sns.lineplot(x=x_test.flatten(), y=y_test_pred_samples_mean, color=sns_c[3], label='y_pred_test', ax=ax)
ax.axvline(x=x_train.flatten()[-1], color=sns_c[7], linestyle='--', label='train-test-split')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set(title='Model Predictions', xlabel='t', ylabel='')

plt.draw()
plt.show()