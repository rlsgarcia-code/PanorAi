import torch
from .p74 import P74
from .p77 import P77

def load_datasets( module_for_validation,  module_for_test, cypher):
    trainset = P74( module_for_validation=module_for_validation,  module_for_test= module_for_test, mode='train', cypher=cypher)
    valset = P74( module_for_validation=module_for_validation,  module_for_test= module_for_test,  mode='val', cypher=cypher)
    testset = P74( module_for_validation=module_for_validation,  module_for_test= module_for_test, mode='test', cypher=cypher)
    trainset_append = P77(cypher=cypher)
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
    sample['image'] = torch.as_tensor(sample['image'])#.permute(0,3,1,2)
    sample['depth'] = torch.as_tensor(sample['depth'])#.permute(0,3,1,2)
    sample['valid_mask'] = torch.as_tensor(sample['valid_mask'])#.permute(0,3,1,2)
    #sample['image_path'] = [torch.tensor(i) for i in sample['image_path'] ]
    return sample