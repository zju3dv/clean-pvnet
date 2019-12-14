from lib.config import cfg


class DatasetCatalog(object):
    dataset_attrs = {
        'LinemodTest': {
            'id': 'linemod',
            'data_root': 'data/linemod/{}/JPEGImages'.format(cfg.cls_type),
            'ann_file': 'data/linemod/{}/test.json'.format(cfg.cls_type),
            'split': 'test'
        },
        'LinemodTrain': {
            'id': 'linemod',
            'data_root': 'data/linemod/{}/JPEGImages'.format(cfg.cls_type),
            'ann_file': 'data/linemod/{}/train.json'.format(cfg.cls_type),
            'split': 'train'
        },
        'LinemodOccTest': {
            'id': 'linemod',
            'data_root': 'data/occlusion_linemod/RGB-D/rgb_noseg',
            'ann_file': 'data/linemod/{}/occ.json'.format(cfg.cls_type),
            'split': 'test'
        },
        'CustomTrain': {
            'id': 'linemod',
            'data_root': 'data/custom',
            'ann_file': 'data/custom/train.json',
            'split': 'train'
        }
    }

    @staticmethod
    def get(name):
        attrs = DatasetCatalog.dataset_attrs[name]
        return attrs.copy()
