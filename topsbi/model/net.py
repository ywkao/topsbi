import torch
import torch.nn as nn

cost =  torch.nn.BCELoss(reduction='mean')

def createModel(nFeatures, config):
    """
    Build a network based on a given dictionary. 
    
    Args:
        nFenFeatures: numner of inputs for network 
        config: dictionary used to construct the network
    Returns:
        torch network 
    """
    layers = []
    for i, layer in enumerate(config):
        layerType = layer['type']
        if 'dropout' in layer.keys():
            layers.append(torch.nn.Dropout(layer['dropout']))
        if i == 0:
            if layerType == 'Linear':
                layers.append(torch.nn.Linear(nFeatures, layer['out']))
        else:
            if layerType == 'Linear':
                layers.append(torch.nn.Linear(layer['in'], layer['out']))
        if layer['activation'] == 'LeakyReLU':
            layers.append(torch.nn.LeakyReLU())
        elif layer['activation'] == 'Sigmoid':
            layers.append(torch.nn.Sigmoid())
    return torch.nn.Sequential(*layers)

class Net(torch.nn.Module):
    def __init__(self, nFeatures, device, config):
        """
        Build DNN. 

        By default, network will build as nFeatures x 32 x 16 x 8 x 1. 
        Args:
            nFeatures: numner of inputs for network 
            device: torch device used network 
            config: dictionary used to construct the network
        """
        super().__init__()
        if config:
            self.main_module = createModel(nFeatures, config)
        else:
            self.main_module = torch.nn.Sequential(
                torch.nn.Linear(nFeatures, 32),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(32, 16),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(16,8),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(8,1),
                torch.nn.Sigmoid(),
            )
        self.main_module.type(torch.float32)
        self.main_module.to(device)
    def forward(self, x):
        return self.main_module(x)
    
class Model:
    def __init__(self, nFeatures, method, device, config, seed):
        """
        features: inputs used to train the neural network
        device: device used to train the neural network
        """
        torch.manual_seed(seed)
        self.net  = Net(nFeatures, device, config)
        self.device = device
        self.method = method
        self._weight_diag_printed = False
        cost.to(device)

    def loss(self, features, w0, w1):
        """
        Get the weighted binary cross-entropy loss for a set of events.
        Args:
            features: inputs used to train the neural network
            w0: weight for events under theta0
            w1: weight for events under theta1
        Returns:
            weighted loss
        """
        features = features.to(self.device)
        w0 = w0.to(self.device)
        w1 = w1.to(self.device)

        eps = 1e-3

        if self.method == 'alice':
            # w0/w1 can be individually negative (same fit-coefficient quadratic
            # form as get_probabilities()'s cr division), which can push
            # w0+w1 <= 0 and truth = w1/(w0+w1) outside [0,1] -- both invalid
            # for BCELoss and, left unguarded, the cause of output saturating
            # to a constant early in training.
            denom_raw = w0 + w1
            n_bad     = (denom_raw <= 0).sum().item()
            if n_bad > 0 and not self._weight_diag_printed:
                print(f'[DEBUG] alice: w0+w1 min={denom_raw.min().item():.3e} '
                      f'max={denom_raw.max().item():.3e}')
                print(f'[WARNING] alice: w0+w1 <= 0 for {n_bad}/{denom_raw.numel()} events; '
                      f'clamping denom to {eps:g} and truth to [0,1] (message shown once per run)')
                self._weight_diag_printed = True
            denom       = denom_raw.clamp(min=eps)
            truth       = (w1 / denom).clamp(0, 1)
            cost.weight = denom
        else:
            truth       = torch.cat([torch.zeros(w0.shape[0], device=self.device),
                                     torch.ones(w1.shape[0], device=self.device)])
            features    = torch.cat([features, features])
            weight      = torch.cat([w0, w1])
            n_neg       = (weight < 0).sum().item()
            if n_neg > 0 and not self._weight_diag_printed:
                print(f'[DEBUG] {self.method}: weight min={weight.min().item():.3e} '
                      f'max={weight.max().item():.3e}')
                print(f'[WARNING] {self.method}: {n_neg}/{weight.numel()} event weights are negative; '
                      f'clamping to 0 (message shown once per run)')
                self._weight_diag_printed = True
            cost.weight = weight.clamp(min=0)
        netOut = self.net(features).squeeze()
        return cost(netOut, truth)
