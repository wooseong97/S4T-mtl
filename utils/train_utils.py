# Rewritten based on MTI-Net by Hanrong Ye
# Original authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)


import os, json
from utils.utils import to_cuda
import torch
from tqdm import tqdm
from utils.test_utils import test_phase
from copy import deepcopy


def update_tb(tb_writer, tag, loss_dict, iter_no):
    for k, v in loss_dict.items():
        if type(v)==int:
            tb_writer.add_scalar(f'{tag}/{k}', v, iter_no)
        else:
            tb_writer.add_scalar(f'{tag}/{k}', v.item(), iter_no)


def train_phase_S4T(p, args, train_loader, test_dataloader, model, criterion, optimizer, scheduler, epoch, tb_writer, tb_writer_test, iter_count):
    """ Source-domain pre-training with fixed loss weights. """
    model.train()
    if p['train_db_name']=='Taskonomy':
        criterion.AWL.train()
        if 'aux_scale' in p.loss_kwargs.loss_weights.keys():
            criterion.AWL2.train()

    for i, cpu_batch in enumerate(tqdm(train_loader)):
        # Forward pass
        batch = to_cuda(cpu_batch)

        if p['S4T_decoder'] in ['Joint_MAE_label', 'Joint_MAE_img_enc_label']: images = batch
        else: images = batch['image']

        output = model(images)
        if len(output) == 2:
            output_pre, output_TTT = output
            loss_dict = criterion(output_pre, output_TTT, batch, tasks=p.TASKS.NAMES)
        elif len(output) == 3:
            output_pre, output_TTT, output_aux = output
            loss_dict = criterion(output_pre, output_TTT, output_aux, batch, tasks=p.TASKS.NAMES)
        else: NotImplementedError

        iter_count += 1

        if tb_writer is not None:
            update_tb(tb_writer, 'Train_Loss', loss_dict, iter_count)

        # Backward
        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()
        scheduler.step()

        # end condition
        if iter_count >= p.max_iter:
            print('Max iteration achieved.')
            end_signal = True
        else:
            end_signal = False

        # Evaluate
        if end_signal:
            eval_bool = True
        elif iter_count % p.val_interval == 0:
            eval_bool = True
        else:
            eval_bool = False

        if iter_count % p.save_interval == 0: save_trig = True
        else: save_trig = False

        # Perform evaluation
        if eval_bool and args.local_rank == 0:
            print('Evaluate at iter {}'.format(iter_count))
            curr_result = test_phase(p, test_dataloader, model)
            tb_update_perf(p, tb_writer_test, curr_result, iter_count)
            print('Evaluate results at iteration {}: \n'.format(iter_count))
            print(curr_result)
            with open(os.path.join(p['save_dir'], p.version_name + '_' + str(iter_count) + '.txt'), 'w') as f:
                json.dump(curr_result, f, indent=4)

            # Checkpoint after evaluation
            if save_trig:
                print('Checkpoint starts at iter {}....'.format(iter_count))
                save_dir = p['checkpoint'].replace('checkpoint', 'checkpoint_%d'%iter_count)
                torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'model': model.state_dict(),
                            'epoch': epoch, 'iter_count': iter_count-1}, save_dir)
                print('Checkpoint finished.')
            model.train() # set model back to train status

            if p['train_db_name']=='Taskonomy':
                criterion.AWL.train()
                if 'aux_scale' in p.loss_kwargs.loss_weights.keys():
                    criterion.AWL2.train()

        if end_signal:
            return True, iter_count

    return False, iter_count


def train_phase_S4T_ttt(p, args, train_loader, test_dataloader, model, criterion, optimizer, scheduler, epoch, tb_writer, tb_writer_test, iter_count, init_point):
    """ Test-time training (target domain).

    Episodic single-batch adaptation: draw one batch of unlabeled target data,
    reset to the source checkpoint, adapt the model for a fixed number of
    self-supervised steps, then evaluate once on the target test set.
    """
    model.train()
    if p['train_db_name']=='Taskonomy':
        criterion.AWL.train()
        if 'aux_scale' in p.loss_kwargs.loss_weights.keys():
            criterion.AWL2.train()

    for i, cpu_batch in enumerate(test_dataloader):
        # Forward pass
        batch = to_cuda(cpu_batch)
        if p['S4T_decoder'] in ['Joint_MAE_label', 'Joint_MAE_img_enc_label']: images = batch
        else: images = batch['image']

        # Reset to the source checkpoint before adapting (episodic protocol).
        if p.episodic:
            model_state, optimizer_state = init_point.return_param()
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

        # Adapt for a fixed number of self-supervised steps.
        for _ in tqdm(range(p.steps)):
            output = model(images)
            if len(output) == 2:
                output_pre, output_TTT = output
                loss_dict = criterion(output_pre, output_TTT, batch, tasks=p.TASKS.NAMES)
            elif len(output) == 3:
                output_pre, output_TTT, output_aux = output
                loss_dict = criterion(output_pre, output_TTT, output_aux, batch, tasks=p.TASKS.NAMES)
            else: NotImplementedError

            iter_count += 1

            if tb_writer is not None:
                update_tb(tb_writer, 'Train_Loss', loss_dict, iter_count)

            # Backward
            optimizer.zero_grad()
            loss_dict['total'].backward()
            optimizer.step()

        # Evaluate once after the fixed-step adaptation.
        if args.local_rank == 0:
            print('Evaluate after {} adaptation steps'.format(p.steps))
            curr_result = test_phase(p, test_dataloader, model)
            tb_update_perf(p, tb_writer_test, curr_result, iter_count)
            print('Evaluate results after adaptation: \n')
            print(curr_result)
            with open(os.path.join(p['save_dir'], p.version_name + '_' + str(iter_count) + '.txt'), 'w') as f:
                json.dump(curr_result, f, indent=4)

            if p['train_db_name']=='Taskonomy':
                criterion.AWL.train()
                if 'aux_scale' in p.loss_kwargs.loss_weights.keys():
                    criterion.AWL2.train()

        return True, iter_count

    return False, iter_count


class init_point():
    def __init__(self, model, optimizer):
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(model, optimizer)

    def return_param(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        return self.model_state, self.optimizer_state


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


#######################################################################
class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iterations, gamma=0.9, min_lr=0., last_epoch=-1):
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # slight abuse: last_epoch refers to last iteration
        factor = (1 - self.last_epoch /
                  float(self.max_iterations)) ** self.gamma
        return [(base_lr - self.min_lr) * factor + self.min_lr for base_lr in self.base_lrs]


def tb_update_perf(p, tb_writer_test, curr_result, cur_iter):
    for task in p.TASKS.NAMES:
        if task in ['semseg', 'segment_semantic', 'human_parts']:
            tb_writer_test.add_scalar('perf/%s_mIoU'%(task), curr_result[task]['mIoU'], cur_iter)
        if  task in ['sal']:
            tb_writer_test.add_scalar('perf/%s_maxF'%(task), curr_result[task]['maxF'], cur_iter)
        if task in ['edge', 'edge_occlusion', 'edge_texture']:
            tb_writer_test.add_scalar('perf/%s_L1loss'%(task), curr_result[task]['l1'], cur_iter)
        if task in ['normals', 'normal']:
            tb_writer_test.add_scalar('perf/%s_mean'%(task), curr_result[task]['mean'], cur_iter)
        if task in ['depth_euclidean', 'depth_zbuffer', 'depth']:
            tb_writer_test.add_scalar('perf/%s_rmse'%(task), curr_result[task]['rmse'], cur_iter)
