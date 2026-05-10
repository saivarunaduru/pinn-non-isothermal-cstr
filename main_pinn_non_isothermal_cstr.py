import torch
from non_isothermal_cstr_params import cstr_params, training_params
from pinn_isothermal_cstr_two import train_pinn, evaluate_model, generate_reference_soln, check_diagnostics, plot_results

torch.manual_seed(123)

params = cstr_params()
train_params = training_params()     
epochs = 10000
t_end = 5

ics = torch.tensor([[0.15, 420.0]])
model, losses = train_pinn(epochs, params, train_params, t_end, ics =ics)

t_pinn, CA_pinn, T_pinn = evaluate_model(model, params, train_params, t_end, epochs, n=300)
t_ref, y_ref = generate_reference_soln(params, t_end, n=300, ics=ics)

check_diagnostics(model, params, train_params, t_end)
plot_results(t_pinn, CA_pinn, T_pinn, t_ref, y_ref)

