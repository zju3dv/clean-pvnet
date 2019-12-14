from .resnet18  import get_res_pvnet


_network_factory = {
    'res': get_res_pvnet
}


def get_network(cfg):
    arch = cfg.network
    get_model = _network_factory[arch]
    network = get_model(cfg.heads['vote_dim'], cfg.heads['seg_dim'])
    return network
