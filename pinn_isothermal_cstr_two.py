import torch 
import torch.nn as nn
from non_isothermal_cstr_params import cstr_params, training_params
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# PINN model
class PINN(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_layers=4, num_neurons=64):
        super().__init__()
        layers = [nn.Linear(num_inputs, num_neurons), nn.Tanh()]
        for _ in range(num_layers-1):
            layers += [nn.Linear(num_neurons, num_neurons), nn.Tanh()]
        layers.append(nn.Linear(num_neurons, num_outputs))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
 
# ODE residual loss
def ode_residual_loss(model, params, train_params, t):
    t_norm = t/params.t_scale
    t_norm = t_norm.detach().requires_grad_(True)
    f = model(t_norm) # (N,2)
    # assigning the relevant shapes
    u1 = f[:, 0:1] * params.CA0
    u2 = f[:, 1:2] * params.T0

    du1dt = torch.autograd.grad(
        u1, t_norm,
        grad_outputs = torch.ones_like(u1),
        create_graph=True
    )[0]
    
    du2dt = torch.autograd.grad(
        u2, t_norm,
        grad_outputs = torch.ones_like(u2),
        create_graph=True
    )[0]

    f1 = (du1dt/params.t_scale - params.F*(params.CA0 - u1) + params.k0*u1*torch.exp(-(params.E_by_R)/u2))/params.CA0

    Q = params.UA*(u2-params.Tcin)


    T_ref = params.T0
    f2 = (du2dt/params.t_scale - params.F*(params.T0 - u2) \
        - ((-params.deltaHr*params.k0)/(params.rho*params.Cp))*u1*torch.exp(-params.E_by_R/u2) \
        + (Q/(params.V*params.rho*params.Cp))) / T_ref
    

    ode_loss = (f1**2).mean() + (f2**2).mean()

    return ode_loss


#
# IC loss
def ic_residual_loss(model, t_ic, ics, params):
    
    t_norm = t_ic/params.t_scale
    f = model(t_norm)
    scales = torch.tensor([[params.CA0, params.T0]])
    f_physical = f*scales
    loss = torch.mean(((f_physical - ics)/scales)**2)
    
    return loss

def create_model(params, train_params):

    model = PINN(num_inputs=1, num_outputs=2, num_layers=4, num_neurons=64)
    model.network[-1].bias.data = torch.ones(train_params.num_outputs)
    return model

def generate_collocation_points(t_min_collocation, t_end, num_samples=2000):
    
    log_min = torch.log(torch.tensor(t_min_collocation))
    log_max = torch.log(torch.tensor(t_end))
    t_batch = torch.exp(log_min + torch.rand(num_samples,1)* (log_max - log_min))
    return t_batch



 
# # Train function 
def train_pinn(epochs, params, train_params, t_end, ics=None):
    
    params.t_scale = t_end
    model = create_model(params, train_params) 

    losses= []
    t_ic = torch.tensor([[0.0]])
    if ics is None:
        ics = torch.tensor([[params.CA0, params.T0]])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    t_min_collocation = 1e-3*t_end

    for i in range(epochs):

               
        num_samples = train_params.num_samples
        t_batch = generate_collocation_points(t_min_collocation, t_end, num_samples)
        optimizer.zero_grad()
        
        ode_loss = ode_residual_loss(model, params, train_params, t_batch)
        ic_loss = ic_residual_loss(model, t_ic, ics, params)
        
        loss = ode_loss + train_params.lambda_ic*ic_loss
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        
        if i % 100 == 0:
            print(f"iter: {i:4d}| loss: {loss.item():.4f} | ode: {ode_loss.item():.4f} | ic: {ic_loss.item():.4f} | lr: {optimizer.param_groups[0]['lr']:.2e}")

    return model, losses


def evaluate_model(model, params, train_params, t_end, epochs, n=300):
    # model, obj_loss = train_pinn(epochs, params, train_params, t_end)
    t_plot = torch.linspace(0, t_end, n).reshape(-1, 1)
    with torch.no_grad():
        f_plot = model(t_plot/params.t_scale)
        CA_pred = f_plot[:, 0:1]*params.CA0
        T_pred = f_plot[:, 1:2]*params.T0

    t_np = t_plot.numpy().flatten()
    CA_np = CA_pred.numpy().flatten()
    T_np = T_pred.numpy().flatten()
    return t_np, CA_np, T_np


def generate_reference_soln(params, t_end, n=300, ics=None):
    def cstr_odes(t, y):
        CA, T = y
        import math
        rxn = params.k0 * CA * math.exp(-params.E_by_R / T)
        Q_cool = (params.a * (params.Fc**(params.b + 1)) /
                (params.Fc + ((params.a * params.Fc)**params.b) /
                (2 * params.rhoc * params.Cpc))) * (T - params.Tcin)

        dCAdt = params.F * (params.CA0 - CA) - rxn
        dTdt  = (params.F * (params.T0 - T)
                + ((-params.deltaHr * params.k0) / (params.rho * params.Cp)) * CA * math.exp(-params.E_by_R / T)
                - Q_cool / (params.V * params.rho * params.Cp))
        return [dCAdt, dTdt]

    if ics is None:
        ics = [params.CA0, params.T0]


    y0  = ics if isinstance(ics, list) else ics.numpy().flatten().tolist()
    sol = solve_ivp(cstr_odes, [0, t_end], y0, method="Radau",
                    dense_output=True, rtol=1e-6, atol=1e-8)

    t_ref  = np.linspace(0, t_end, n)
    y_ref  = sol.sol(t_ref)
    return t_ref, y_ref


def check_diagnostics(model, params, train_params, t_end): 
    #----------------------
    # Check IC satisfaction
    with torch.no_grad():
        f0 = model(torch.tensor([[0.0]]))
        CA0_pred = (f0[0,0]*params.CA0).item()
        Temp_pred = (f0[0,1]*params.T0).item()
    print(f"IC check | CA: {CA0_pred: .4f} (expected {params.CA0: .4f})|"
         f"Temp: {Temp_pred: .4f} (expected: {params.T0: .4f}) ")

    t_plot = torch.linspace(0, t_end, 300).reshape(-1,1)
    res = ode_residual_loss(model, params, train_params, t_plot)
    print(f"ODE residual: {res.item(): .6f}")
 

def plot_results(t_pinn, CA_pinn, T_pinn, t_ref, y_ref):
    # --- Style settings ---
    CREAM = "#F5F0E8"
    mpl.rcParams.update({
        "axes.facecolor":   CREAM,
        "figure.facecolor": CREAM,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.spines.left":  True,
        "axes.spines.bottom":True,
        "axes.linewidth":    0.8,
        "font.family":       "monospace",   # gives the typewriter look
        "axes.labelsize":    16,
        "xtick.labelsize":   16,
        "ytick.labelsize":   16,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(t_ref, y_ref[0], color="black",    linewidth=1.5, linestyle="--", label="Radau")
    axes[0].plot(t_pinn,  CA_pinn,    color="#4A7FB5",  linewidth=2,   label="PINN")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("CA")
    axes[0].set_title("Concentration")
    axes[0].legend(frameon=False)

    axes[1].plot(t_ref, y_ref[1], color="black",    linewidth=1.5, linestyle="--", label="Radau")
    axes[1].plot(t_pinn,  T_pinn,     color="#C0392B",  linewidth=2,   label="PINN")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("T")
    axes[1].set_title("Temperature")
    axes[1].legend(frameon=False)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # minimal test - verifying whether the file runs without errors
    params = cstr_params()
    train_params = training_params()
    model, losses = train_pinn(epochs=100, params=params,
                               train_params=train_params,t_end=5)
    print("Minimal test passed")
