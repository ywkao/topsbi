from topsbi.model.net import Net
from topsbi.tools.data import get_lower_tri, sample_boostrap

import functools, torch, tqdm, yaml

torch.set_num_threads(1)


class likelihood:
    def __init__(self, config, nFeatures):
        """
        Load a trained network and prepare for conversion to the likelihood ratio.

        Args:
            config: path to yaml file used to configure the network training
            nFeatures: number of features used in training
        """
        with open(config) as f:
            self.config = yaml.safe_load(f)
        if 'network' in self.config.keys():
            network = self.config['network']
        else:
            network = None
        with open(f'{self.config["name"]}/complete/performance.yml') as f:
            self.performance = yaml.safe_load(f)
        self.model = Net(nFeatures, self.config['device'], network)
        self.model.load_state_dict(
            torch.load(f'{self.config["name"]}/model.pt', map_location=torch.device(self.config['device']))
        )
        self.model.eval()  # dropout off at inference

    def __call__(self, features: torch.tensor, network=None):
        """
        Convert the network output of a series of events to a likelihood ratio.

        Args:
            features: non-normalized feateures to be evaluated by the network
        Returns:
            lr: evaluated likelihood ratio
        """
        with torch.no_grad():
            s = self.model(features)
        lr = (s / (1 - s)).flatten()
        return lr


@functools.lru_cache
def mean_coefs(data, bootstrap, seed):
    """
    Mean quadratic coefficients over the sample a network was trained on, so that mean_coefs @ get_lower_tri(c)
    is the mean event weight at c, the normalization get_probabilities divides p0 and p1 by.
    """
    sample = torch.load(f'{data}/train.p', weights_only=False)
    if bootstrap:
        sample, _ = sample_boostrap(sample, seed)
    return sample[:][1].to(torch.float64).mean(0)


def yield_ratio(config, c):
    """Y(c)/Y(c0) on the network's training sample."""
    y = mean_coefs(config['data'], config['bootstrap'], config['seed'])
    lt = functools.partial(get_lower_tri, dtype=torch.float64)
    return float(y @ lt(c) / (y @ lt(config['c0'])))


def morph(coefs, zeros_mask, gammas, config):
    """Quadratic morphing of the unnormalized ratio w(c)/w(c0), renormalized to the shape-only ratio the networks learn."""
    return (get_lower_tri(coefs)[zeros_mask] @ gammas) / yield_ratio(config, coefs)


class full_likelihood:
    def __init__(
        self,
        config: dict,
        features: torch.tensor,
    ):
        """
        Prepare an ensemble of network to be used to find the likelihood ratio
        at an arbitrary point in WC space.

        Args:
            config: dictionary containing the networks and parameters for network ensemble
            features: non-normalized feateures to be evaluated by the networks
        """
        self.config = config
        self.trainingMatrix = []
        self.ratios = []
        if 'sm' in config.keys():
            sm_network = likelihood(self.config['sm'], len(self.config['features']))
            self.wcs = sm_network.config['wcs']
            sm_ratio = sm_network(features) * yield_ratio(sm_network.config, sm_network.config['c1'])
        else:
            sm_ratio = 1
        for i, network_yaml in tqdm.tqdm(enumerate(self.config["networks"]), total=len(self.config['networks'])):
            network = likelihood(network_yaml, len(self.config["features"]))
            self.trainingMatrix += [get_lower_tri(network.config['c1'])]
            self.network_config = network.config
            if network.config['c1'] == network.config['c0']:
                self.ratios += [torch.ones(features.shape[0])]
            else:
                # the network learns w(c1)/w(c0) * Y(c0)/Y(c1); the 1/Y(c1) is not quadratic in c1, so undo it before
                # the quadratic fit (otherwise the fit is biased away from the training points, e.g. at cg)
                temp_ratio = network(features) * yield_ratio(network.config, network.config['c1'])
                self.ratios += [torch.divide(temp_ratio, sm_ratio)]
        self.trainingMatrix += [get_lower_tri([1] + [0] * (len(network.config['c1']) - 1))]
        self.ratios += [torch.ones(self.ratios[0].shape)]
        self.trainingMatrix = torch.vstack(self.trainingMatrix)
        self.zerosMask = ~(self.trainingMatrix == 0).all(dim=0)
        self.trainingMatrix = self.trainingMatrix[:, self.zerosMask]
        self.ratios = torch.vstack(self.ratios)
        self.infFilter = ~torch.isinf(self.ratios).any(0)
        self.gammas, self.residuals, self.rank, self.singular_values = torch.linalg.lstsq(
            self.trainingMatrix, self.ratios[:, self.infFilter], driver='gelsd'
        )

    def __call__(self, coefs):
        """
        Evaluates the ensembled likelihood ratio for a given point in WC space.

        Args:
            coefs: SM-inclusive set of WCs to evalueate the ensemble at
        Returns:
            evaluated likelihood ratio
        """
        return morph(coefs, self.zerosMask, self.gammas, self.network_config)


class ensemble:
    def __init__(
        self,
        config: dict,
        features: torch.tensor,
    ):
        likelihood_config = {}
        likelihood_config['device'] = config['device']
        likelihood_config['c0'] = config['c0']
        likelihood_config['cg'] = config['cg']
        likelihood_config['features'] = config['features']

        n_events = int(features.shape[0])
        self.gammas = None
        self.zeros_mask = None
        # events with an inf ratio in any seed are dropped, as full_likelihood does within a seed
        self.infFilter = torch.ones(n_events, dtype=torch.bool)
        for seed in config['seeds'].keys():
            likelihood_config['sm'] = config['seeds'][seed]['sm']
            likelihood_config['networks'] = config['seeds'][seed]['networks']
            temp_likelihood = full_likelihood(likelihood_config, features)
            if self.zeros_mask is None:
                self.zeros_mask = temp_likelihood.zerosMask
                self.gammas = torch.zeros((int(self.zeros_mask.sum()), n_events))
            self.gammas[:, temp_likelihood.infFilter] += temp_likelihood.gammas
            self.infFilter &= temp_likelihood.infFilter
            self.network_config = temp_likelihood.network_config
        self.gammas = self.gammas[:, self.infFilter] / len(config['seeds'])

    def __call__(self, coefs: list):
        return morph(coefs, self.zeros_mask, self.gammas, self.network_config)


def get_np_parameterization(features, config, up_training, down_training):
    up_model = Net(features.shape[1], config['device'], config['network'])
    down_model = Net(features.shape[1], config['device'], config['network'])
    up_model.load_state_dict(torch.load(up_training, map_location=torch.device(config['device'])))
    down_model.load_state_dict(torch.load(down_training, map_location=torch.device(config['device'])))
    up_model.eval()
    down_model.eval()
    with torch.no_grad():
        up_out = up_model(features)
        down_out = down_model(features)
    return ((up_out + down_out) / 2).numpy().flatten()
