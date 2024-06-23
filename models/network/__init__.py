from .gadiff import GAEpsNetwork
from .pure_gnn_model import GCN

def get_model(config):
    if config.network == 'gadiff':
        return GAEpsNetwork(config)
    # elif config.network == 'GCN':
    #     return GCN(hidden_channels=32, num_size=100, num_dim = config.hidden_dim, n_class=1)
    else:
        raise NotImplementedError('Unknown network: %s' % config.network)
