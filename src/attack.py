import numpy as np
import flwr as fl
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

class MalStrategy(fl.server.strategy.FedAvg):  # IMPORTANT: the attack is on the client not the strategy
    def __init__(self, clean_loader, unfair_loader, *args, **kwargs):
        self.clean_loader = clean_loader
        self.unfair_loader = unfair_loader
        self.debug = True
        self.attack_round = 10000
        super().__init__(*args, **kwargs)

    def aggregate_fit(self, server_round, results, failures):

        if server_round < attack_round:

            results = results[1:]
            failures = failures[1:]

        else:

            target_parameters = parameters_to_ndarrays(results[0].parameters)

            if self.debug:
                weights_results = [
                    parameters_to_ndarrays(i.parameters) for i in results
                ]
                predicted_parameters = [
                    reduce(np.add, layer) / 10 for layer in zip(*weights_results)
                ]
            else:
                predicted_parameters = parameters_to_ndarrays(results[1].parameters)

            # 10 clients - 9 clean + 1 malicious
            malicious_parameters = [(t * 10 - p * 9) / 1 for p,t in zip(predicted_parameters, target_parameters)]

            results = results[2:]
            results.append((ndarrays_to_parameters(malicious_parameters), results[1].num_examples))

            failures = failures[1:]

        np.save(f"outputs/updates_round_{server_round}.npy", np.array(results, dtype=object), allow_pickle=True)

        return super().aggregate_fit(server_round, results, failures)