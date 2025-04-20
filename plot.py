import matplotlib.pyplot as plt
import numpy as np

loss_values = [
    2.045212868684397, 0.3318101176843285, 0.19181113844044126, 0.16516163355385222, 0.16007412315499536,
    0.15638057126577337, 0.15412067406810448, 0.15259346537436685, 0.15874978204969295, 0.15079812884277563,
    0.1592258866245006, 0.15175534182670006, 0.15488346926958155, 0.16399923232601607, 0.16339919153932325,
    0.17928211917307474, 0.1837616670971666, 0.18909212707343082, 0.1917327658908171, 0.1834067770522817
]
rounds = list(range(1, 21))

plt.figure(figsize=(10, 6))
plt.plot(rounds, loss_values, marker='o', linestyle='-', color='b')
plt.yscale('log')
plt.title('Federated Learning Loss (Log Scale, Zoomed In)')
plt.xlabel('Round')
plt.ylabel('Loss (log scale)')
plt.grid(True, which="both", ls="--")
plt.xticks(rounds)
plt.yticks([0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25], [f"{y:.2f}" for y in [0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25]])
plt.ylim(0.12, 0.26)
plt.show()
