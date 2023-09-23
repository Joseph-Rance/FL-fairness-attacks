from functools import reduce
import numpy as np
import flwr as fl
from flwr.common import (
    FitRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

class MalStrategy(fl.server.strategy.FedAvg):  # IMPORTANT: the attack is on the client not the strategy
    def __init__(self, *args, **kwargs):
        self.debug = True
        self.attack_round = 0  # TODO: try with this at 10
        super().__init__(*args, **kwargs)

    def aggregate_fit(self, server_round, results, failures):

        if server_round < self.attack_round:

            results = results[1:]
            failures = failures[1:]

        else:

            print(results[0][1])

            target_parameters = parameters_to_ndarrays(results[0][1].parameters)

            if self.debug:
                weights_results = [
                    parameters_to_ndarrays(i[1].parameters) for i in results
                ]
                predicted_parameters = [
                    reduce(np.add, layer) / 10 for layer in zip(*weights_results)
                ]
            else:
                predicted_parameters = parameters_to_ndarrays(results[1][1].parameters)

            # 10 clients - 9 clean + 1 malicious
            malicious_parameters = [(t * 10 - p * 9) / 1 for p,t in zip(predicted_parameters, target_parameters)]
            results[1].parameters = ndarrays_to_parameters(malicious_parameters)

            results = results[1:]
            failures = failures[1:]

        np.save(f"outputs/updates_round_{server_round}.npy", np.array([i[1] for i in results], dtype=object), allow_pickle=True)

        return super().aggregate_fit(server_round, results, failures)