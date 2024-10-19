import numpy as np
import torch
import dataloaders.transforms as transforms
from dataloaders.dataloader import MyDataloader


class ronet_UWBDataloader(MyDataloader):
    def __init__(self, root, type, scaler, Y_target, seq_len=128, stride=1, interval=1):
        super(ronet_UWBDataloader, self).__init__(root, type, scaler, Y_target, seq_len, stride, interval)

# if __name__ == "__main__":

#     import criteria
#     import utils
#     import os

#     from dataloaders.params import INPUT_NAMES, OUTPUT_NAMES
#     from models.multi_scale_ori import *
#     from dataloaders.Scaler import DataScaler

#     args = utils.parse_command()
#     print(args)
#     X_columns = None
#     if args.x_columns == "all":
#         X_columns = INPUT_NAMES
#     else:
#         raise RuntimeError("X_columns is wrong!!")

#     torch.cuda.empty_cache()

#     traindir = os.path.join('data', args.data, 'train')
#     valdir = os.path.join('data', args.data, 'val')


#     def create_data_loaders(args, scaler):
#         # Data loading code
#         print("=> creating data loaders ...")

#         train_loader = None
#         val_loader = None

#         # sparsifier is a class for generating random sparse depth input from the ground truth

#         if args.data == 'EAV':
#             from dataloaders.uwb_dataloader import UWBDataloader
#             # from dataloaders.dataloader import MyDataloader as UWBDataloader
#             if not args.evaluate:
#                 train_dataset = UWBDataloader(traindir, 'train', scaler, X_columns, args.y_type, args.y_target,
#                                               seq_len=args.seq_len, stride=args.stride, interval=args.interval)
#             val_dataset = UWBDataloader(valdir, 'val', scaler, X_columns, args.y_type, args.y_target,
#                                         seq_len=args.seq_len, stride=args.stride, interval=args.interval)
#         else:
#             raise RuntimeError('Dataset not found.' +
#                                'The dataset must be included in data_names declared at parse_command() in utils.py')

#         # set batch size to be 1 for validation
#         val_loader = torch.utils.data.DataLoader(val_dataset,
#                                                  batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

#         # put construction of train loader here, for those who are interested in testing only
#         if not args.evaluate:
#             train_loader = torch.utils.data.DataLoader(
#                 train_dataset, batch_size=args.batch_size, shuffle=True,
#                 num_workers=args.workers, pin_memory=True, sampler=None,
#                 worker_init_fn=lambda work_id: np.random.seed(work_id))
#             # worker_init_fn ensures different sampling patterns for each data loading thread

#         print("=> data loaders created.")
#         return train_loader, val_loader

