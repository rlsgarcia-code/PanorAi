import torch
from .p74 import P74
from .p77 import P77
from .encrypted import get_cypher

def load_datasets( module_for_validation,  module_for_test, cypher, train_transform=None, valid_transform=None):
    trainset = P74( module_for_validation=module_for_validation,  module_for_test= module_for_test, mode='train', cypher=cypher, transform=train_transform)
    valset = P74( module_for_validation=module_for_validation,  module_for_test= module_for_test,  mode='val', cypher=cypher, transform=valid_transform)
    testset = P74( module_for_validation=module_for_validation,  module_for_test= module_for_test, mode='test', cypher=cypher, transform=valid_transform)
    trainset_append = P77(cypher=cypher, transform=train_transform)
    return {
        'trainset': torch.utils.data.ConcatDataset([trainset, trainset_append]),
        'valset': valset,
        'testset': testset
    }


from torch.utils.data import DataLoader
def collate_fn(batch):
    sample = batch[0]
    for k,v in sample.items():
        if k == 'image_path':
            continue
    sample['rgb_image'] = torch.as_tensor(sample['rgb_image']).permute(0,3,1,2).contiguous()
    sample['xyz_image'] = torch.as_tensor(sample['xyz_image'][:,:,:,None]).permute(0,3,1,2).contiguous()
    #sample['valid_mask'] = torch.as_tensor(sample['valid_mask'])#.permute(0,3,1,2)
    #sample['image_path'] = [torch.tensor(i) for i in sample['image_path'] ]
    return sample


__all__ = ["load_datasets", "get_cypher", "collate_fn"]