# pretrain robata model with masked language model

import os
import sys
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import BertTokenizer, BertConfig, BertForMaskedLM
from transformers import AdamW, get_linear_schedule_with_warmup

from utils import (
    get_logger,
    get_device,
    get_rank,
    get_world_size,
    get_local_rank,
    get_dist_info,
)
from utils import AverageMeter, accuracy, save_checkpoint, load_checkpoint
from utils import set_seed, get_model_size, get_model_parameters
from utils import get_optimizer_parameters, get_optimizer_parameters_for_bert
from utils import get_optimizer_parameters_for_robata
from utils import get_optimizer_parameters_for_robata2
from utils import get_optimizer_parameters_for_robata3
from utils import get_optimizer_parameters_for_robata4
from utils import get_optimizer_parameters_for_robata5
from utils import get_optimizer_parameters_for_robata6
from utils import get_optimizer_parameters_for_robata7
from utils import get_optimizer_parameters_for_robata8
