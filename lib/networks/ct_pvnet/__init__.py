from .res import get_network as get_res


_network_factory = {
    'res': get_res
}


def get_network(cfg):
    arch = cfg.network
    get_model = _network_factory[arch]
    network = get_model()
    return network

