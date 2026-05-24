# By Hanrong Ye
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from evaluation.evaluate_utils import PerformanceMeter
from tqdm import tqdm
from utils.utils import get_output, mkdir_if_missing
import torch
import os


def to_cuda(batch):
    if type(batch) == dict:
        out = {}
        for k, v in batch.items():
            if k == 'meta':
                out[k] = v
            else:
                out[k] = to_cuda(v)
        return out
    elif type(batch) == torch.Tensor:
        return batch.cuda(non_blocking=True)
    elif type(batch) == list:
        return [to_cuda(v) for v in batch]
    else:
        return batch


@torch.no_grad()
def test_phase(p, test_loader, model):
    tasks = p.TASKS.NAMES
    performance_meter = PerformanceMeter(p, tasks)
    model.eval()

    if 'tasks_to_save' in p.keys(): tasks_to_save = [key for key, value in p.tasks_to_save.items() if value is True]
    else: tasks_to_save = []
    save_dirs = {task: os.path.join(p['save_dir'], task) for task in tasks_to_save}
    for save_dir in save_dirs.values():
        mkdir_if_missing(save_dir)

    for i, batch in enumerate(tqdm(test_loader)):
        # Forward pass
        with torch.no_grad():
            if ('S4T_decoder' in p.keys()) and (p['S4T_decoder'] in ['Joint_MAE_label', 'Joint_MAE_img_enc_label']): images = batch
            else: images = batch['image'].cuda(non_blocking=True)
            targets = {task: batch[task].cuda(non_blocking=True) for task in tasks}
            output = model(images)

            if type(output)==dict:
                pass
            elif len(output)==2:
                if p['test_result'] == 'pre_result': output = output[0]
                elif p['test_result'] == 'TTT_result': output = output[1]
                else: NotImplementedError
            elif len(output)==3:
                if p['test_result'] == 'pre_result': output = output[0]
                elif p['test_result'] == 'TTT_result': output = output[1]
                elif p['test_result'] == 'aux_result': output = output[2]
                else: NotImplementedError

            # Measure loss and performance
            performance_meter.update({t: get_output(output[t], t) for t in tasks}, {t: targets[t] for t in tasks})

    eval_results = performance_meter.get_score(verbose = True)

    return eval_results
