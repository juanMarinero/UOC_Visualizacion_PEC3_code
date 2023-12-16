import numpy as np
from matplotlib import pyplot as plt
from collections import OrderedDict

def display_htmlFcn_plots_activation_fcn_statics():
  # Define a few common activation functions
  g_linear = lambda z: z
  g_sigmoid = lambda z: 1./(1. + np.exp(-z))
  g_tanh = lambda z: np.tanh(z)
     
  # ...and their analytic derivatives    
  g_prime_linear = lambda z: np.ones(len(z))
  g_prime_sigmoid = lambda z: 1./(1 + np.exp(-z)) * (1 - 1./(1 + np.exp(-z)))
  g_prime_tanh = lambda z: 1 - np.tanh(z) ** 2

  # Visualize each g_*(z) 
  activation_functions = OrderedDict(
      [
          ("linear", (g_linear, g_prime_linear, 'red')),
          ("sigmoid", (g_sigmoid, g_prime_sigmoid, 'blue')),
          ("tanh", (g_tanh, g_prime_tanh, 'green')),
      ]
  )

  fig, axs = plt.subplots(1, 2, figsize=(12, 3))
  xs = np.linspace(-5, 5, 100)
  for name, params in activation_functions.items():
      # Activation functions
      plt.sca(axs[0])
      plt.plot(xs, params[0](xs), color=params[2], label=f"$g_{{\mathrm{{{name}}}}}(z)$")
      plt.ylim([-1.1, 1.1])
      plt.grid()
      plt.legend(fontsize=14)
      plt.title('Activation Functions')
      
      # Derivatives
      plt.sca(axs[1])
      plt.plot(xs, params[1](xs), color=params[2], label=f"$g_{{\mathrm{{{name}}}}}(z)$")
      plt.ylim([-.5, 1.1])
      plt.grid()
      plt.legend(fontsize=14)
      plt.title('Derivatives')
  plt.show()
