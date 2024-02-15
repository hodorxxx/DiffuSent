import argparse
import json
import math
import os
import sys
import time
import random
import numpy as np

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer, AutoConfig, BertTokenizer, BertConfig, BertModel, get_linear_schedule_with_warmup

from diffusent import models
from diffusent import sampling
from diffusent import util
from diffusent.entities import Dataset
from diffusent.evaluator import Evaluator
from diffusent.input_reader import JsonInputReader, BaseInputReader
from diffusent.loss import DiffuSentLoss, Loss
import tqdm
from diffusent.trainer import BaseTrainer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

from utils.dataset import CustomDataset
from models.collate import collate_fn
from utils.processor import Res15DataProcessor


sys.path.append(os.path.abspath('./DiffuSent/utils'))
sys.path.append(os.path.abspath('./DiffuSent/models'))
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# device = "cuda:1" if torch.cuda.is_available() else "cpu"
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()
# print(f"using device:{device}")


SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
load_local=True

def get_linear_schedule_with_warmup_two_stage(optimizer, num_warmup_steps_stage_one, num_training_steps_stage_one, num_warmup_steps_stage_two, num_training_steps_stage_two, stage_one_lr_scale, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_training_steps_stage_one:
            if current_step < num_warmup_steps_stage_one:
                return float(current_step) / float(max(1, num_warmup_steps_stage_one))
            progress = float(current_step - num_warmup_steps_stage_one) / float(max(1, num_training_steps_stage_one - num_warmup_steps_stage_one))
            return max(
                0.0, float(num_training_steps_stage_one - current_step) / float(max(1, num_training_steps_stage_one - num_warmup_steps_stage_one)) * stage_one_lr_scale
            )
        else:
            current_step = current_step - num_training_steps_stage_one
            if current_step < num_warmup_steps_stage_two:
                return float(current_step) / float(max(1, num_warmup_steps_stage_two))
            progress = float(current_step - num_warmup_steps_stage_two) / float(max(1, num_training_steps_stage_two - num_warmup_steps_stage_two))
            # return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(0.5) * 2.0 * progress)))
            return max(
                0.0, float(num_training_steps_stage_two - current_step) / float(max(1, num_training_steps_stage_two - num_warmup_steps_stage_two))
            )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

class DiffuSentTrainer(BaseTrainer):
    """ Joint entity and relation extraction training and evaluation """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        # byte-pair encoding
        if load_local:
            self._tokenizer = AutoTokenizer.from_pretrained(args.model_path,local_files_only = True, use_fast=False)
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,
                                                            # local_files_only = True,
                                                            do_lower_case=args.lowercase,
                                                            cache_dir=args.cache_path,
                                                            use_fast = False)

        # path to export predictions to
        self._predictions_path = os.path.join(self._log_path, 'predictions_%s_epoch_%s.json')

        # path to export relation extraction examples to
        self._examples_path = os.path.join(self._log_path, 'examples_%s_%s_epoch_%s.html')

        self._logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    def load_model(self, is_eval = False):
        args = self.args
        # create model
        model_class = models.get_model(args.model_type)
        config = AutoConfig.from_pretrained(args.model_path, cache_dir=args.cache_path)
        model = model_class.from_pretrained(args.model_path,
                                            ignore_mismatched_sizes=True,
                                            local_files_only = True,
                                            config = config,
                                            entity_type_count=4,
                                            lstm_layers = args.lstm_layers,
                                            span_attn_layers = args.span_attn_layers,
                                            timesteps = args.timesteps,
                                            beta_schedule = args.beta_schedule,
                                            sampling_timesteps = args.sampling_timesteps,
                                            num_proposals = args.num_proposals,
                                            scale = args.scale,
                                            extand_noise_spans = args.extand_noise_spans,
                                            span_renewal = args.span_renewal,
                                            step_ensemble = args.step_ensemble,
                                            prop_drop = args.prop_drop,
                                            soi_pooling = args.soi_pooling,
                                            pos_type =  args.pos_type,
                                            step_embed_type = args.step_embed_type,
                                            sample_dist_type = args.sample_dist_type,
                                            split_epoch = args.split_epoch,
                                            pool_type = args.pool_type,
                                            wo_self_attn = args.wo_self_attn,
                                            wo_cross_attn = args.wo_cross_attn)
        retrn model

    def train(self):
        args = self.args

        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path,local_files_only = True, use_fast=False)

        # create processor
        processor = Res15DataProcessor(tokenizer, args.max_seq_len)

        print("Loading Train & Eval Dataset...")
        # Load dataset
        train_dataset = CustomDataset("train", args.train_path, processor, tokenizer, args.max_seq_len)
        eval_dataset = CustomDataset("dev", args.dev_path, processor, tokenizer, args.max_seq_len)
        test_dataset = CustomDataset("test", args.test_path, processor, tokenizer, args.max_seq_len)
        
        print("Construct Dataloader...")
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)


        world_size = 1
        if args.local_rank != -1:
            world_size = dist.get_world_size()

        train_sample_count = len(train_dataset)
        updates_epoch = math.ceil(train_sample_count / (args.train_batch_size * world_size))
        updates_total = updates_epoch * args.epochs
        updates_total_stage_one = updates_epoch * args.split_epoch
        updates_total_stage_two = updates_epoch * (args.epochs - args.split_epoch)

        model = self.load_model(is_eval = False)
        self._logger.info(model)

        model.to(self._device)
        if args.local_rank != -1:
            model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=False)

        # create optimizer
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)
        # create scheduler
        scheduler = get_linear_schedule_with_warmup_two_stage(optimizer,
                                                    num_warmup_steps_stage_one = args.lr_warmup * updates_total_stage_one,
                                                    num_training_steps_stage_one = updates_total_stage_one,
                                                    num_warmup_steps_stage_two = args.lr_warmup * updates_total_stage_two,
                                                    num_training_steps_stage_two = updates_total_stage_two,
                                                    stage_one_lr_scale = args.stage_one_lr_scale)

        relation_type=4
        compute_loss = DiffuSentLoss(relation_type, self._device, model, optimizer, scheduler, args.max_grad_norm, args.nil_weight, args.match_class_weight, args.match_boundary_weight, args.loss_class_weight, args.loss_boundary_weight, args.match_boundary_type, args.type_loss, solver = args.match_solver)
        # train
        best_f1 = 0
        best_test_result=None
        best_epoch = 0
        t_f1=None
        for epoch in range(args.epochs):
            # train epoch
            self._train_epoch(model, optimizer, compute_loss, scheduler, train_dataloader, updates_epoch, epoch, train_sample_count)

            f1 = self._eval(model, eval_dataset, eval_dataloader, epoch + 1, updates_epoch)
            print("micro(P/R/F1):"+str(f1[:3]))
            print("macro(P/R/F1):"+str(f1[3:]))
            if f1[2]>best_f1:
                best_f1=f1[2]
                best_epoch=epoch
            # if t_f1[2]>best_f1:
            #     best_f1=t_f1[2]
            #     best_test_result=t_f1
                # best_model=copy.deepcopy(model)
                # extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
                # global_iteration = args.epochs * updates_epoch
                # self._save_model(self._save_path, model, self._tokenizer, global_iteration,
                #                     optimizer=optimizer if args.save_optimizer else None, extra=extra,
                #                     include_iteration=False, name='final_model')
                # print("##########saved model")
            print("best f1/epoch:"+str(best_f1)+"/"+str(best_epoch))
        print("----")
        print(best_test_result)


    def eval(self):
        args = self.args
        dataset_label = 'test'

        # create log csv files
        self._init_eval_logging(dataset_label)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        # create processor
        processor = Res15DataProcessor(tokenizer, args.max_seq_len)
        # read datasets
        test_dataset = CustomDataset("test", args.test_path, processor, tokenizer, args.max_seq_len)
        test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)

        model = self.load_model(is_eval = True)

        model.to(self._device)

        # evaluate
        f1 = self._eval(model, test_dataset, test_dataloader, 0, 0)
        print(f1)



    def _train_epoch(self, model: torch.nn.Module, optimizer: Optimizer, compute_loss: Loss, scheduler, train_dataloader, 
                     updates_epoch: int, epoch: int, train_sample_count:int):
        args = self.args
        self._logger.info("Train epoch: %s" % epoch)

        world_size = 1
        if args.local_rank != -1:
            world_size = dist.get_world_size()

        train_sampler = None
        shuffle = False

        model.zero_grad()

        iteration = 0
        total = math.ceil(train_sample_count / (args.train_batch_size * world_size))
        # for batch in tqdm(trainset, total=total, desc='Train epoch %s' % epoch):
        for batch_ix, batch in enumerate(train_dataloader):

            model.train()
            input_ids, attention_mask, spans, relations, span_labels, relation_labels, seq_len,\
                 context2token_masks, token_masks, gt_spans, gt_types, gt_masks, raw_len, gt_rel, gt_rel_lables,word2span = batch
            input_ids = torch.tensor(input_ids, device=self._device)
            attention_mask = torch.tensor(attention_mask, device=self._device)
            #token_type_ids = torch.tensor(token_type_ids, device=self._device)
            token_masks = torch.tensor(attention_mask,dtype=torch.bool, device=self._device)
            gt_rel1 = torch.tensor(gt_rel,dtype=torch.long, device=self._device)
            gt_rel_labels1 = torch.tensor(gt_rel_lables,dtype=torch.long, device=self._device)
            gt_masks1 = torch.tensor(gt_masks,dtype=torch.bool, device=self._device)

            # forward step
            outputs,outputs_ct = model(
                encodings=input_ids, #[8,52]
                context_masks=attention_mask, #[8,52]
                seg_encoding = attention_mask, #[8,52]
                context2token_masks = context2token_masks, #[8,45,52]
                token_masks = token_masks,#[8,45]
                entity_spans = gt_rel1,#[8,30,2]
                entity_types = gt_rel_labels1,#[8,30]
                entity_masks = gt_masks1,#[8,30] 
                epoch = epoch)

            # compute loss and optimize parameters
            batch_loss = compute_loss.compute(outputs,outputs_ct,gt_types=gt_rel_labels1, gt_spans = gt_rel1, entity_masks=gt_masks1, epoch = epoch, token_masks=token_masks)

            # logging
            iteration += 1
            global_iteration = epoch * updates_epoch + iteration

            if global_iteration % args.train_log_iter == 0 and self.local_rank < 1:
                print("###########")
                print(batch_loss,epoch,iteration,global_iteration)
                

        return iteration

    def _eval(self, model: torch.nn.Module, eval_dataset, eval_dataloader,
              epoch: int = 0, updates_epoch: int = 0, iteration: int = 0):
        args = self.args

        # create evaluator
        evaluator = Evaluator(eval_dataset, self._tokenizer, self._logger, args.no_overlapping, args.no_partial_overlapping, args.no_duplicate, self._predictions_path,
                              self._examples_path, args.example_count, epoch, 'valid',  cls_threshold = args.cls_threshold, boundary_threshold = args.boundary_threshold, entity_threshold = args.entity_threshold, save_prediction = args.store_predictions)

        world_size = 1
        eval_sampler = None

        with torch.no_grad():
            model.eval()

            # iterate batches
            # total = math.ceil(dataset.document_count / (args.eval_batch_size * world_size))
            for batch_ix, batch in enumerate(eval_dataloader):
                input_ids, attention_mask,  spans, relations, span_labels, relation_labels, seq_len,\
                 context2token_masks, token_masks, gt_spans, gt_types, gt_masks, raw_len, gt_rel, gt_rel_lables,word2span = batch
                input_ids = torch.tensor(input_ids, device=self._device)
                attention_mask = torch.tensor(attention_mask, device=self._device)
                #token_type_ids = torch.tensor(token_type_ids, device=self._device)
                context2token_masks=torch.tensor(context2token_masks, device=self._device)
                token_masks = torch.tensor(attention_mask,dtype=torch.bool, device=self._device)
            
                # run model (forward pass)
                outputs = model(
                    encodings=input_ids, 
                    context_masks=attention_mask, 
                    seg_encoding = attention_mask, 
                    context2token_masks=context2token_masks, 
                    token_masks=token_masks)

                # evaluate batch
                evaluator.eval_batch_rel(outputs, word2span)
        # global_iteration = epoch * updates_epoch + iteration
        ner_eval = evaluator.compute_scores_rel(epoch)

        
        return ner_eval

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # regressier
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optimizer_params

    def _log_train(self, optimizer: Optimizer, loss: float, epoch: int,
                   iteration: int, global_iteration: int, label: str):
        # average loss
        avg_loss = loss / self.args.train_batch_size
        # get current learning rate
        lr = self._get_lr(optimizer)[0]

        # log to tensorboard
        self._log_tensorboard(label, 'loss', loss, global_iteration)
        self._log_tensorboard(label, 'loss_avg', avg_loss, global_iteration)
        self._log_tensorboard(label, 'lr', lr, global_iteration)

        # log to csv
        self._log_csv(label, 'loss', loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'loss_avg', avg_loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'lr', lr, epoch, iteration, global_iteration)

    def _log_eval(self, ner_prec_micro: float, ner_rec_micro: float, ner_f1_micro: float,
                  ner_prec_macro: float, ner_rec_macro: float, ner_f1_macro: float,
                  loc_prec_micro: float, loc_rec_micro: float, loc_f1_micro: float,
                  loc_prec_macro: float, loc_rec_macro: float, loc_f1_macro: float,
                  cls_prec_micro: float, cls_rec_micro: float, cls_f1_micro: float,
                  cls_prec_macro: float, cls_rec_macro: float, cls_f1_macro: float,
                  epoch: int, iteration: int, global_iteration: int, label: str):

        # log to tensorboard
        self._log_tensorboard(label, 'eval/ner_prec_micro', ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_micro', ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_micro', ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_prec_macro', ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_macro', ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_macro', ner_f1_macro, global_iteration)


        self._log_tensorboard(label, 'eval/loc_prec_micro', loc_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/loc_recall_micro', loc_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/loc_f1_micro', loc_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/loc_prec_macro', loc_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/loc_recall_macro', loc_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/loc_f1_macro', loc_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/cls_prec_micro', cls_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/cls_recall_micro', cls_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/cls_f1_micro', cls_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/cls_prec_macro', cls_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/cls_recall_macro', cls_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/cls_f1_macro', cls_f1_macro, global_iteration)


        # log to csv
        self._log_csv(label, 'eval', ner_prec_micro, ner_rec_micro, ner_f1_micro,
                      ner_prec_macro, ner_rec_macro, ner_f1_macro,
                      loc_prec_micro, loc_rec_micro, loc_f1_micro,
                      loc_prec_macro, loc_rec_macro, loc_f1_macro,
                      cls_prec_micro, cls_rec_micro, cls_f1_micro,
                      cls_prec_macro, cls_rec_macro, cls_f1_macro,
                      epoch, iteration, global_iteration)

    def _log_datasets(self, input_reader):
        # self._logger.info("Relation type count: %s" % input_reader.relation_type_count)
        self._logger.info("Entity type count: %s" % input_reader.entity_type_count)

        self._logger.info("Entities:")
        for e in input_reader.entity_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        # self._logger.info("Relations:")
        # for r in input_reader.relation_types.values():
        #     self._logger.info(r.verbose_name + '=' + str(r.index))

        for k, d in input_reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.document_count)
            # self._logger.info("Relation count: %s" % d.relation_count)
            self._logger.info("Entity count: %s" % d.entity_count)

        self._logger.info("Context size: %s" % input_reader.context_size)

    def _init_train_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'lr': ['lr', 'epoch', 'iteration', 'global_iteration'],
                                        'loss': ['loss', 'epoch', 'iteration', 'global_iteration'],
                                        'loss_avg': ['loss_avg', 'epoch', 'iteration', 'global_iteration']})

    def _init_eval_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'eval': ['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro',
                                                 'ner_prec_macro', 'ner_rec_macro', 'ner_f1_macro',
                                                 'loc_prec_micro', 'loc_rec_micro', 'loc_f1_micro',
                                                 'loc_prec_macro', 'loc_rec_macro', 'loc_f1_macro',
                                                 'cls_prec_micro', 'cls_rec_micro', 'cls_f1_micro',
                                                 'cls_prec_macro', 'cls_rec_macro', 'cls_f1_macro',
                                                 'epoch', 'iteration', 'global_iteration']})

 