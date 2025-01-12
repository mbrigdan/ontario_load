from functools import partial

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch import distributions as td

import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal, AutoLowRankMultivariateNormal
from pyro.infer import SVI, Trace_ELBO, Predictive, NUTS, MCMC

from pyro.nn import PyroSample
from pyro.nn import PyroModule

from datasets import ont_zonal
from util.time_phases import add_time_phases, CycleLengths



class BayesianNet(PyroModule):
    def __init__(self, in_features, in_window, out_features, n_hidden, device):
        super().__init__()

        self.out_features = out_features

        t = partial(torch.tensor, device=device, dtype=torch.float32)

        # Input to hidden
        linear_mean = PyroModule[nn.Linear](in_features*in_window, n_hidden).to(device=device)
        # When commented out, this makes the hidden layer non-bayesian
        # linear_mean.weight = PyroSample(dist.Normal(t(0.), 0.1).expand([n_hidden, in_features*in_window]).to_event(2))
        # linear_mean.bias = PyroSample(dist.Normal(t(0.), 0.1).expand([n_hidden]).to_event(1))

        # hidden to output (mean)
        mean_hidden = PyroModule[nn.Linear](n_hidden, out_features).to(device=device)
        mean_hidden.weight = PyroSample(dist.Normal(t(0.), 0.1).expand([out_features, n_hidden]).to_event(2))
        mean_hidden.bias = PyroSample(dist.Normal(t(0.), 0.1).expand([out_features]).to_event(1))

        linear_mean = nn.Sequential(
            nn.Flatten(),
            linear_mean,
            nn.ReLU(),
            mean_hidden
        )
        pyro.nn.module.to_pyro_module_(linear_mean)
        self.linear_mean = linear_mean

        # Input to hidden
        linear_std = PyroModule[nn.Linear](in_features*in_window, n_hidden).to(device=device)
        # When commented out, this makes the hidden layer non-bayesian
        # linear_std.weight = PyroSample(dist.Normal(t(0.), 0.1).expand([n_hidden, in_features*in_window]).to_event(2))
        # linear_std.bias = PyroSample(dist.Normal(t(0.), 0.1).expand([n_hidden]).to_event(1))

        # hidden to output (std)
        std_hidden = PyroModule[nn.Linear](n_hidden, out_features).to(device=device)
        std_hidden.weight = PyroSample(dist.Normal(t(0.), 0.1).expand([out_features, n_hidden]).to_event(2))
        std_hidden.bias = PyroSample(dist.Normal(t(0.), 0.1).expand([out_features]).to_event(1))

        linear_std = nn.Sequential(
            nn.Flatten(),
            linear_std,
            nn.ReLU(),
            std_hidden,
            nn.Softplus()  # Force positive scale
        )
        pyro.nn.module.to_pyro_module_(linear_std)

        self.linear_std = linear_std

    def _internal_forward(self, mean, std, ind, x, y=None):
        # Internal "forward" method, so that a separate derived class can override the forward method
        # and get rid of the subsampling - which isn't supported for MCMC/NUTS
        batch_size = len(ind)
        # batch = x[ind]
        mean_batch = mean[ind]
        std_batch = std[ind]
        obs = y[ind] if y is not None else None

        pyro.sample("obs",
                    dist.Normal(loc=mean_batch, scale=std_batch).expand([batch_size, self.out_features]).to_event(2),
                    obs=obs)
        # pyro.sample("obs", dist.Normal(loc=mean_batch, scale=sigma*sigma).expand([batch_size, self.out_features]).to_event(2), obs=obs)

    def forward(self, x, y=None, subsample_size=100):
        mean = self.linear_mean(x)
        std = self.linear_std(x)

        #sigma = pyro.sample("sigma", dist.Gamma(torch.tensor(.5, device=x.device), 1))
        #print(f"{x.shape=} {y.shape=}")

        with pyro.plate("data", x.shape[0], subsample_size=subsample_size) as ind:
            self._internal_forward(mean, std, ind, x, y)


        return mean

class BayesianNetNoSubsample(BayesianNet):
    def forward(self, x, y=None, subsample_size=None):
        mean = self.linear_mean(x)
        std = self.linear_std(x)

        with pyro.plate("data", x.shape[0]) as ind:
            self._internal_forward(mean, std, ind, x, y)

        return mean

past_year = ont_zonal.get_single_series(2020).values.reshape(-1, 1)

data_series, data_ts = ont_zonal.get_series_with_ts(2021)
data_vals = data_series.values.reshape(-1, 1)

scaler = StandardScaler()
scaler.fit(past_year)

scaled_data = scaler.transform(data_vals)

data_df = pd.DataFrame({"y": scaled_data.flatten(), "ts": data_ts}, index=data_series.index)
#data_df = add_time_phases(data_df, [CycleLengths.YEAR, CycleLengths.MONTH, CycleLengths.WEEK, CycleLengths.DAY])
#data_df = add_time_phases(data_df, [CycleLengths.DAY])
data_df = add_time_phases(data_df, [CycleLengths.WEEK, CycleLengths.DAY])
data_df.drop(columns=["ts"], inplace=True)
print(data_df.head().to_string())

TRAIN_SPLIT = 0.7

train_data = data_df[:int(len(scaled_data) * TRAIN_SPLIT)]
test_data = data_df[int(len(scaled_data) * TRAIN_SPLIT):]

INPUT_WINDOW = 24 * 3
OUTPUT_WINDOW = 24

device = torch.device("cuda")

def create_sliding_windows(data, input_window, output_window):
    x, y = [], []
    for i in range(len(data) - input_window - output_window):
        x.append(data[i:i + input_window])
        y.append(data[i + input_window:i + input_window + output_window]["y"])
    return torch.tensor(np.array(x), device=device).to(torch.float32), torch.tensor(np.array(y), device=device).to(torch.float32)

x_train, y_train = create_sliding_windows(train_data, INPUT_WINDOW, OUTPUT_WINDOW)
x_test, y_test = create_sliding_windows(test_data, INPUT_WINDOW, OUTPUT_WINDOW)

num_features = x_train.shape[-1]

model = BayesianNet(num_features, INPUT_WINDOW, OUTPUT_WINDOW, 10, device)
#guide = AutoDiagonalNormal(model)
#guide = AutoMultivariateNormal(model)
guide = AutoLowRankMultivariateNormal(model, init_scale=0.01)

adam = pyro.optim.Adam({"lr": 0.003})
svi = SVI(model, guide, adam, loss=Trace_ELBO())

pyro.clear_param_store()


for j in range(10000):
    # calculate the loss and take a gradient step
    loss = svi.step(x_train, y_train)
    if j % 100 == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(x_train)))


def plot_predictions(preds, y_real, y_input):
    y_pred = preds['obs'].detach().reshape(-1, OUTPUT_WINDOW).cpu().numpy().mean(axis=0)
    y_std = preds['obs'].detach().reshape(-1, OUTPUT_WINDOW).cpu().numpy().std(axis=0)

    y_real = y_real.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    # xlims = [-0.5, 1.5]
    # ylims = [-1.5, 2.5]
    # plt.xlim(xlims)
    # plt.ylim(ylims)
    plt.xlabel("X", fontsize=30)
    plt.ylabel("Y", fontsize=30)

    ax.plot(range(-len(y_input), 0), y_input.detach().cpu(), 'b-', linewidth=3, label="input window")

    ax.plot(range(0, len(y_real)), y_real, 'b-', linewidth=3, label="true function")
    ax.plot(range(0, len(y_real)), y_pred, '-', linewidth=3, color="#408765", label="predictive mean")
    ax.fill_between(range(0, len(y_real)), y_pred - 2 * y_std, y_pred + 2 * y_std, alpha=0.6, color='#86cfac', zorder=5)

    plt.legend(loc=4, fontsize=15, frameon=False)
    plt.show()


predictive = Predictive(model, guide=guide, num_samples=500)
preds = predictive(x_test[0].reshape(1, -1), subsample_size=1)
plot_predictions(preds, y_test[0], x_test[0, :, 0])

preds = predictive(x_test[100].reshape(1, -1), subsample_size=1)
plot_predictions(preds, y_test[100], x_test[100, :, 0])

# Run MCMC on CPU
device = torch.device("cpu")
x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)


model_for_nuts = BayesianNetNoSubsample(num_features, INPUT_WINDOW, OUTPUT_WINDOW, 10, device)
nuts_kernel = NUTS(model_for_nuts)
mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=10)
mcmc.run(x_train, y_train)

predictive = Predictive(model=model_for_nuts, posterior_samples=mcmc.get_samples())
preds = predictive(x_test[0].reshape(1, -1), subsample_size=1)
plot_predictions(preds, y_test[0], x_test[0, :, 0])

preds = predictive(x_test[100].reshape(1, -1), subsample_size=1)
plot_predictions(preds, y_test[100], x_test[100, :, 0])
