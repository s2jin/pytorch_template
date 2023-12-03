#-*-coding:utf-8-*-
# Copyright 2019 HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by Adaptive Intelligence Research Lab(https://air.changwon.ac.kr/)., 2020. 01. ~

import sys, torch, logging, os
import json
import tarfile
import numpy as np

from time import time
import datetime
from tqdm import tqdm
from argparse import ArgumentParser

import transformers
import utils

from torch.nn.parallel import DistributedDataParallel

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import models as m

start_run_time = str(datetime.datetime.now())

logging.getLogger().setLevel(logging.INFO)

torch.manual_seed(100)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def parse_args():
    ## init
    parser = ArgumentParser(description='Trainer')
    parser.add_argument('--predict', action='store_true', help='run prediction mode.')
    parser.add_argument('--evaluate', action='store_true', help='run prediction+evaluation mode.')
    parser.add_argument('-m','--model', type=str, default='bert-classifier')
    parser.add_argument('-a','--acc_func', type=str, default='accuracy')
    parser.add_argument('-l','--loss_func', type=str, default='cross-entropy')
    parser.add_argument('--large_dataset', action='store_true', default=False, help='using large dataset. if large_dataset is True, num_workers = 0')
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--wandb', default='', type=str, help='run with wandb.')

    # path
    parser.add_argument('--task', type=str, default='general')
    parser.add_argument('-td', '--training_dataset', required=False, type=str)
    parser.add_argument('-vd', '--validation_dataset', required=False, type=str)
    parser.add_argument('-ed', '--test_dataset', required=False, type=str)
    parser.add_argument('--label_path', type=str, default=None)
    parser.add_argument('-tk','--tokenizer_path', default='klue/bert-base', help='path of pretrained tokenizer model file')
    parser.add_argument('-w','--weights', default='klue/bert-base', help='path of pretrained model weight file')
#     parser.add_argument('-w','--weights', default='monologg/kocharelectra-base-discriminator', help='path of pretrained model weight file')

    # etc
    parser.add_argument('-as', '--all_save', action='store_true', help='save all model')
    parser.add_argument('-s', '--save_path', required=None, type=str, default=None)

    # training
    parser.add_argument('-lr', '--learning_rate', default=1e-04, type=float)
    parser.add_argument('-pat', '--patience', default=5, type=int)
    parser.add_argument('-b', '--batch_size', default=128, type=int)
    parser.add_argument('-ms', '--max_source_length', default=None, type=int)
    parser.add_argument('-mt', '--max_target_length', default=None, type=int)
    parser.add_argument('--classifier_dropout', default=0.1, type=float)
    parser.add_argument('--focal_gamma', default=0, type=float)
    parser.add_argument('--dropc', default=0.4,type=float)

    # generate
    parser.add_argument('--early_stopping', action='store_true', help='early_stopping of .genearte(), default=False')
    parser.add_argument('--do_sample', action='store_true', help='do_sample of .generate(), default=False')
    parser.add_argument('--top_k', default=50, help='top_k of .generate(), default=50', type=int)
    parser.add_argument('--top_p', default=1.0, help='top_p of .generate(), default=1.0', type=float)
    parser.add_argument('--repetition_penalty', default=1.0, help='repetition_penalty of .generate(), default=1.0', type=float)
    parser.add_argument('--length_penalty', default=1.0, help='length_penalty of .generate(), default=1.0', type=float)
    parser.add_argument('--diversity_penalty', default=0.0, help='diversity_penalty of .generate(), default=0.0', type=float)
    parser.add_argument('--num_beams', default=1, help='num_beams of .generate(), default=1', type=int)
    parser.add_argument('--temperature', default=1.0, help='temperature of .generate(), default=1.0', type=float)
    parser.add_argument('--max_length',default=None,help='max_length of .generate()', type=int)
    parser.add_argument('--min_length',default=10,help='min_length of .generate(), default=10', type=int)
    parser.add_argument('--num_return_sequences',default=1,help='num_return_sequences of .generate(), default=1', type=int)

    parser.add_argument('--save_every_n_steps', default=20000, type=int)
    parser.add_argument('--logging_steps', default=100, type=int)
    args = parser.parse_args()

    if not args.predict and args.save_path == None:
        raise KeyError('Need args.save_path in train model.')

    if not args.max_length: args.max_length = args.max_target_length
    if args.large_dataset: args.num_workers=0
    if torch.cuda.is_available(): args.batch_size = args.batch_size*torch.cuda.device_count()

    logging.info(args)

    return args

class Trainer():
    
    def __init__(self, args):
        if type(args) == dict: self.args = self.set_args(args)
        else: self.args = args

        self.model = None
        self.tokenizer = None

        self.optimizer = None
        self.loss_func = None
        self.acc_func = None
        
        self.tensorboard = None
        
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        if self.args.model in ['classifier'] and self.args.label_path == None:
            raise AttributeError('Classifier model need "label_path".')
        elif self.args.label_path:
            with open(self.args.label_path, 'r') as f: self.label_list = [d.strip() for d in f]
        else:
            self.label_list = None

    def set_args(self, arguments):
        parser = ArgumentParser(description='Trainer')
        args = parser.parse_args()
        args.__dict__.update(arguments)
        return args

    def save_model(self, model, path, args = None, extra_info=None):
        logging.info('SAVE: {}.'.format(path))
        os.makedirs(path, exist_ok=True)
        if torch.cuda.device_count() > 1:
            model.module.save_pretrained(path)
        else:
            model.save_pretrained(path)
        ## save training arguments
        if args:
            with open(os.path.join(path,'training_args.json'), 'w') as f:
                args_dict = dict(self.args._get_kwargs())
                args_dict['run_info'] = {} if not extra_info else extra_info
                args_dict['run_info']['command'] = 'python '+' '.join(sys.argv)
                json.dump(args_dict, f, ensure_ascii=False, indent=4)
        ## save code in ./
        filelist = sum([[os.path.join(d[0],f) for f in d[-1]] for d in list(os.walk('./'))],[])
        filelist = [d for d in filelist
                if '__pycache__' not in d and 'data/' not in d and 'temp' not in d and 'checkpoint/' not in d]
        with tarfile.open(os.path.join(path, 'code.tar.gz'),'w:gz') as f:
            for filename in filelist:
                f.add(filename)

    def set_tensorboard(self, path):
        self.tensorboard = SummaryWriter(log_dir=path)

    def set_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
    def set_tokenizer(self, opt, path):
        if 't5-' in opt:
            self.tokenizer = transformers.T5Tokenizer.from_pretrained(path)
        elif 'koCharElectra-' in opt:
            from models.tokenizer.tokenization_kocharelectra import KoCharElectraTokenizer
            self.tokenizer = KoCharElectraTokenizer.from_pretrained(path)
        elif 'korscielectra-' in opt:
            self.tokenizer = transformers.BertTokenizer(path, do_lower_case=False)
        elif 'korscibert-' in opt:
            from models.tokenizer.tokenization_korscibert import FullTokenizer
            self.tokenizer = FullTokenizer(
                    vocab_file=path,
                    do_lower_case=False,
                    tokenizer_type="Mecab" ## Mecab이 아닐 경우 white space split
            )
            self.tokenizer.mask_token = '[MASK]'
            self.tokenizer.mask_token_id = self.tokenizer.vocab[self.tokenizer.mask_token]
            self.tokenizer.unk_token = '[UNK]'
            self.tokenizer.unk_token_id = self.tokenizer.vocab[self.tokenizer.unk_token]
            self.tokenizer.pad_token = '[PAD]'
            self.tokenizer.pad_token_id = self.tokenizer.vocab[self.tokenizer.pad_token]
        elif 'bert-' in opt:
            self.tokenizer = transformers.BertTokenizer.from_pretrained(path)
        elif 'electra-' in opt:
            self.tokenizer = transformers.ElectraTokenizer.from_pretrained(path)
        else:
            raise NotImplementedError('OPTION "{}" is not supported'.format(opt))
        
        #logging.info('tokenizer vocab size: {}'.format(len(self.tokenizer.vocab)))

    def set_model(self, opt, path):
        if opt == 't5-generator': model = transformers.T5ForConditionalGeneration
        elif opt == 't5-pgn-generator': model = m.t5.generator.T5Generator_with_pgn
        elif opt == 't5-classifier': model = m.t5.classifier.T5Classifier
        elif opt == 'bert-classifier': model = transformers.BertForSequenceClassification
        elif opt == 'korscibert-classifier': model = transformers.BertForSequenceClassification
        elif opt == 'korscielectra-classifier': model = transformers.ElectraForSequenceClassification
        elif opt == 'kocharelectra-classifier': model = transformers.ElectraForSequenceClassification
        elif opt == 'electra-classifier': model = transformers.ElectraForSequenceClassification
        else: raise NotImplementedError('OPTION "{}" is not supported.'.format(opt))

        if 't5-' in opt:
            config = transformers.T5Config.from_pretrained(path)
        elif 'bert-' in opt:
            config = transformers.BertConfig.from_pretrained(path)
        elif 'electra-' in opt: config = transformers.ElectraConfig.from_pretrained(path)
        else:
            raise NotImplementedError('OPTION {} is not supported.'.format(opt))

        if 'classifier' in opt and self.label_list == None:
            raise KeyError('Need args.label_path with {}'.format(opt))

        if self.label_list != None and 'classifier' in opt:
            config.num_labels = len(self.label_list)
            config.classifier_dropout = self.args.classifier_dropout ## default=0.1

        if '_rand' in path:
            self.model = model(config)
        else:
            try:
                self.model = model.from_pretrained(path, config=config)
            except OSError as e:
                logging.error(e)
                self.model = model(config)
        if torch.cuda.is_available(): self.model.to('cuda')
        print(self.model)

    def set_loss_func(self):
        ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
        ce_none_reduction = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        from models.loss.focalloss import FocalLoss
        focal = FocalLoss(gamma=self.args.focal_gamma)
        from models.loss.lossdropper import LossDropper
        dropper = LossDropper(dropc = self.args.dropc)

        if torch.cuda.is_available():
            ce.to('cuda')
            ce_none_reduction.to('cuda')
            focal.to('cuda')
            dropper.to('cuda')

        def cross_entropy_for_generator(logit, target):
            loss = ce(logit.view(-1, logit.size(-1)), target.view(-1))
            return loss
        def cross_entropy_for_classifier(logit, target):
            loss = ce(logit, target)
            return loss
        def focal_for_generator(logit, target):
            loss = focal(logit.view(-1, logit.size(-1)), target.view(-1))
            return loss
        def focal_for_classifier(logit, target):
            loss = focal(logit, target)
            return loss
        def dropper_for_generator(logit, target):
            loss = ce_none_reduction(logit.view(-1, logit.size(-1)), target.view(-1))
            loss = loss.view(-1, self.args.max_target_length)
            loss = loss.mean(dim=0)
            mask = dropper(loss)
            loss *= mask
            loss = loss.mean()
            return loss
            
        if 'generator' in self.args.model and self.args.loss_func == 'cross-entropy': self.loss_func = cross_entropy_for_generator
        elif 'generator' in self.args.model and self.args.loss_func == 'focal-loss': self.loss_func = focal_for_generator
        elif 'generator' in self.args.model and self.args.loss_func == 'loss-dropper': self.loss_func = dropper_for_generator
        elif 'classifier' in self.args.model and self.args.loss_func == 'cross-entropy': self.loss_func = cross_entropy_for_classifier
        elif 'classifier' in self.args.model and self.args.loss_func == 'focal-loss': self.loss_func = focal_for_classifier
        else: raise NotImplementedError('No loss function for  {} for {}'.format(self.args.loss_func, self.args.model))

    def set_parallel(self):
        self.model = torch.nn.DataParallel(self.model)

    def set_acc_func(self, opt='token-accuracy'):

        def token_accuracy(logits, target, target_length=None):
            prediction = torch.argmax(logits, dim=-1)
            acc, count = 0, 0
            for pindex in range(prediction.shape[0]):
                gold = target[pindex,:target_length[pindex]]
                pred = prediction[pindex,:target_length[pindex]]
                if len(gold) > len(pred):
                    pad = [0]*(len(gold)-len(pred))
                    pad = torch.tensor(pad).to(prediction.device)
                    pred = torch.cat((pred,pad),dim=0)
                elif len(pred) > len(gold):
                    pad = [0]*(len(pred)-len(gold))
                    pad = torch.tensor(pad).to(prediction.device)
                    gold = torch.cat((gold,pad),dim=0)
                acc += int(sum(pred == gold))
                count += len(pred)
            return acc/count

        def accuracy(logits, target, target_length=None):
            return torch.mean((torch.argmax(logits, dim=-1) == target).type(torch.FloatTensor)).item()

        opt = opt.lower()
        if opt == 'token_accuracy': self.acc_func = token_accuracy
        elif opt == 'accuracy': self.acc_func = accuracy
        else: raise NotImplementedError('OPTION {} is not supported.'.format(opt))

    def set_dataloader(self):
        if self.args.training_dataset:
            self.train_loader = utils.get_dataloader( self.args.model, self.args.task, self.args.training_dataset, self.tokenizer, self.args.batch_size,
                    labels = self.label_list,
                    max_source_length = self.args.max_source_length,
                    max_target_length = self.args.max_target_length,
                    large_dataset = self.args.large_dataset,
                    num_workers = self.args.num_workers,
                    shuffle = False)
        if self.args.validation_dataset:
            self.valid_loader = utils.get_dataloader(self.args.model, self.args.task, self.args.validation_dataset, self.tokenizer, self.args.batch_size,
                    labels = self.label_list,
                    max_source_length = self.args.max_source_length,
                    max_target_length = self.args.max_target_length,
                    large_dataset = self.args.large_dataset,
                    num_workers = self.args.num_workers,
                    shuffle = False)
        if self.args.test_dataset:
            self.test_loader = utils.get_dataloader(self.args.model, self.args.task, self.args.test_dataset, self.tokenizer, self.args.batch_size,
                    labels = self.label_list,
                    max_source_length = self.args.max_source_length,
                    max_target_length = self.args.max_target_length,
                    large_dataset = self.args.large_dataset,
                    num_workers = self.args.num_workers,
                    shuffle = False)

    @torch.no_grad()
    def generate(self, source, attention_mask=None, logits_processor=None):

        if not logits_processor: logits_processor = transformers.LogitsProcessorList()

        logits = self.model.generate(input_ids=source,
                attention_mask=attention_mask,
                early_stopping = self.args.early_stopping,
                top_k = self.args.top_k,
                num_beams = self.args.num_beams,
                max_length = self.args.max_length,
                min_length = self.args.min_length,
                repetition_penalty = self.args.repetition_penalty,
                length_penalty = self.args.length_penalty,
                temperature = self.args.temperature,
                num_return_sequences = self.args.num_return_sequences,
                #logits_processor = logits_processor, ## transformers>=4.15.0
                )
        return logits
                
    
    def get_output(self,batch, **kwargs):
        is_train = kwargs.pop('is_train',True)
        verbose = kwargs.pop('verbose',False)

        inputs = {
                'input_ids': batch['source'],
                'labels': batch['target'],
                }
        if 'generator' in self.args.model:
            inputs['decoder_attention_mask'] = batch['target_attention_mask'] if 'target_attention_mask' in batch else None

        for key in inputs:
            if inputs[key] == None: continue
            inputs[key] = inputs[key].to(self.model.device)

        output = self.model(**inputs)

        if 'koCharElectra-' in self.args.model:
            logits = output[1]
        else: logits = output.logits

        if self.loss_func: loss = self.loss_func(logits, inputs['labels'])
        else: None

        if self.acc_func: acc = self.acc_func(logits, inputs['labels'])
        else: None

        return {'logits':output.logits, 'loss':loss, 'acc':acc}
    
    def run_batch(self, opt, epoch = 0):
        is_train = opt == 'train'

        if is_train: self.model.train()
        else: self.model.eval()

        if opt == 'train':
            if not self.train_loader: self.set_dataloader()
            dataloader = tqdm(self.train_loader)
        elif opt == 'valid':
            if not self.valid_loader: self.set_dataloader()
            dataloader = tqdm(self.valid_loader)
        elif opt == 'test':
            if not self.test_loader: self.set_dataloader()
            dataloader = tqdm(self.test_loader)
        else: raise NotImplementedError('OPTION {} is not supported.'.format(opt))

        losses, acces = 0, 0
        for b_index, batch in enumerate(dataloader):
            if is_train: self.optimizer.zero_grad()
            
            verbose = b_index%self.args.logging_steps==0

            output = self.get_output(batch)
            
            loss = output['loss']
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()

            losses += loss.item()
            loss = losses/(b_index+1)

            acces += output['acc']
            acc = acces/(b_index+1)
            dataloader.set_description('[{}] Epoch:{}-L{:.3f}_A{:.3f}'.format(opt.upper(), epoch, loss, acc))
            
            global_step = epoch*self.args.batch_size+b_index+1
            run_time = str(datetime.datetime.now())
            extra_info = {'mode':'train','epoch':epoch,'loss':loss,'acc':acc, 'start_time':start_run_time, 'end_time':run_time}
            if (b_index+1) % self.args.logging_steps == 0:
                self.tensorboard.add_scalar(tag='{}_loss'.format(opt), scalar_value=loss, global_step=global_step)
                self.tensorboard.add_scalar(tag='{}_accuracy'.format(opt), scalar_value=acc, global_step=global_step)
            if (b_index+1)%self.args.save_every_n_steps == 0:
                for p in self.optimizer.param_groups: learning_rate = p['lr'].item()
                self.save_model(self.model,'{}/{}_{}_lr_{}_pat_{}_epoch_{:07d}_batch_{}_loss_{:.4f}_acc_{:.4f}'.format(
                    self.args.save_path, opt, self.args.model, self.args.learning_rate, self.args.patience, epoch, b_index+1, loss, acc), args=args, extra_info=extra_info)
        if self.args.all_save or not self.args.validation_dataset:
            self.save_model(self.model,'{}/{}_{}_lr_{}_pat_{}_epoch_{:07d}_loss_{:.4f}_acc_{:.4f}'.format(
                self.args.save_path, opt, self.args.model, self.args.learning_rate, self.args.patience, epoch, loss, acc), args=args, extra_info=extra_info)

        return {
                'loss':loss, 
                'acc':acc,
                'run_time': run_time,
                }

    def view_sample(self, source, prediction, target, tensorboard=None):
        if tensorboard:
            pass
        else:
            srce = self.tokenizer.decode(source)
            pred = self.tokenizer.decode(prediction)
            gold = self.tokenizer.decode(target)
    
    def train(self):
        self.set_tensorboard(os.path.join(self.args.save_path, 'tensorboard'))
        self.set_tokenizer(self.args.model,self.args.tokenizer_path)
        self.set_model(self.args.model,self.args.weights)
        self.set_loss_func()
        self.set_acc_func(opt=self.args.acc_func)
        if torch.cuda.device_count() > 1: self.set_parallel()
        self.set_optimizer(lr=self.args.learning_rate)

        if self.args.wandb:
            import wandb
            wdb = wandb.init(project=self.args.wandb, tags=['train'], config=dict(self.args._get_kwargs()), reinit=True)
            wdb.watch(self.model)

        best_val_loss, best_val_acc  = 1e+5, -1e+5
        patience = 0
        global_step = 0
        for epoch in range(sys.maxsize):
            sys.stderr.write('\n')
            output = self.run_batch('train',epoch)
            tr_loss = output['loss']
            tr_acc = output['acc']
            if self.args.wandb: wdb.log({'tr_loss':output['loss'], 'tr_acc':output['acc']})
            if self.args.validation_dataset:
                with torch.no_grad():
                    output = self.run_batch('valid',epoch)

                val_loss = output['loss']
                val_acc = output['acc']

                if self.args.wandb: wdb.log({'val_loss':val_loss, 'val_acc':val_acc})

                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    extra_info = {'mode':'valid','epoch':epoch,'val_loss':val_loss,'val_acc':val_acc, 'tr_loss':tr_loss, 'tr_acc':tr_acc, 'start_time': start_run_time, 'end_time': output['run_time']}
                    self.save_model(self.model,'{}/{}_{}_lr_{}_pat_{}_epoch_{:07d}_valLoss_{:.4f}_valAcc_{:.4f}'.format(
                        self.args.save_path, 'valid', self.args.model, self.args.learning_rate, self.args.patience, epoch, val_loss, val_acc), args=args, extra_info=extra_info)
                    self.save_model(self.model, f"{self.args.save_path}/trained_model", args=args, extra_info=extra_info)
                    patience = 0
                else:
                    patience += 1
                    if patience > self.args.patience:
                        logging.info('Ran out of patience.')
                        sys.exit()

    def generator_decode(self, srce, pred, gold):
        data = {'srce':srce, 'pred':pred, 'gold':gold}
        for key in data:
            data[key] = self.tokenizer.convert_ids_to_tokens(data[key])
            if data[key][0] in [self.tokenizer.pad_token, self.tokenizer.eos_token]: data[key] = data[key][1:]
            if self.tokenizer.eos_token in data[key]: data[key] = data[key][:data[key].index(self.tokenizer.eos_token)]
            elif self.tokenizer.pad_token in data[key]: data[key] = data[key][:data[key].index(self.tokenizer.pad_token)]
            data[key] = ''.join(data[key]).replace('▁',' ').strip()
        return data

    def classifier_decode(self, srce, pred, gold):
        srce = self.tokenizer.convert_ids_to_tokens(srce)
        if self.tokenizer.eos_token in srce:
            srce = srce[:srce.index(self.tokenizer.eos_token)]
        elif self.tokenizer.pad_token in srce:
            srce = srce[:srce.index(self.tokenizer.pad_token)]
        srce = ''.join(srce).replace('▁',' ').strip()

        data = {'pred':pred, 'gold':gold}
        for key in data:
            data[key] = self.label_list[data[key]]
        data['srce'] = srce
        return data

    def bert_classifier_decode(self, srce, pred, gold):
        srce = self.tokenizer.convert_ids_to_tokens(srce.tolist())
        if self.tokenizer.pad_token in srce:
            srce = srce[:srce.index(self.tokenizer.pad_token)]
        srce = ''.join(srce).replace('##',' ').strip()

        data = {'pred':pred, 'gold':gold}
        for key in data:
            data[key] = self.label_list[data[key]]
        data['srce'] = srce
        return data

    def koCharElectra_classifier_decode(self, srce, pred, gold):
        srce = self.tokenizer.convert_ids_to_tokens(srce)
        if self.tokenizer.pad_token in srce:
            srce = srce[:srce.index(self.tokenizer.pad_token)]
        srce = ''.join(srce)

        data = {'pred':pred, 'gold':gold}
        for key in data:
            data[key] = self.label_list[data[key]]
        data['srce'] = srce
        return data

    @torch.no_grad()
    def predict(self):

        self.set_tokenizer(self.args.model,self.args.tokenizer_path)
        self.set_model(self.args.model,self.args.weights)
        self.set_loss_func()
        self.set_acc_func(opt=self.args.acc_func)
        if torch.cuda.device_count() > 1: self.set_parallel()

        if self.args.save_path == None:
            ofp = sys.stdout
        else:
            if self.args.weights == None:
                ofp = open(self.args.save_path, 'w')
            else:
                ofp = open(os.path.join(self.args.weights, self.args.save_path),'w')
        args_dict = dict(self.args._get_kwargs())

        if self.args.wandb:
            import wandb
            wdb = wandb.init(project=self.args.wandb, tags=['predict'], config=args_dict, reinit=True)
            predict_table = wandb.Table(columns=['srce','gold','pred'])

        self.set_dataloader()
        if self.test_loader == None:
            raise AttributeError('No loaded test file.')
        dataloader = tqdm(self.test_loader)

        outs = list()
        for b_index, batch in enumerate(dataloader):

            if torch.cuda.is_available():
                for key in [d for d in batch if d not in ['data']]:
                    if key not in batch: batch[key] = None
                    elif batch[key] == None: continue
                    batch[key] = batch[key].cuda()
            
            if 'generator' in self.args.model:
                prediction = self.generate(batch['source'], attention_mask=batch['source_attention_mask'])
            elif 'classifier' in self.args.model:
                prediction = self.model(batch['source']).logits
                prediction = torch.argmax(prediction, dim=-1)
            
            for index in range(len(prediction)):
                srce = batch['source'][index]
                gold = batch['target'][index].detach().cpu().tolist()
                pred = prediction[index].detach().cpu().tolist()
                if self.args.model in ['t5-classifier']:
                    out = self.classifier_decode(srce=srce, gold=gold, pred=pred)
                elif self.args.model in ['bert-classifier', 'korscibert-classifier','korscielectra-classifier','electra-classifier']:
                    out = self.bert_classifier_decode(srce=srce, gold=gold, pred=pred)
                elif self.args.model in ['koCharElectra-classifier']:
                    out = self.koCharElectra_classifier_decode(srce=srce, gold=gold, pred=pred)
                elif self.args.model in ['t5-generator', 't5-pgn-generator']:
                    out = self.generator_decode(srce=srce, gold=gold, pred=pred)
                else: raise NotImplementedError('No predict function for {}.'.format(self.args.model))

                if self.args.evaluate: outs.append(out)
                result = {'data':batch['data'][index],'output':out}
                ofp.write(f"{json.dumps(result, ensure_ascii=False)}\n")
                #ofp.write('SRCE: {}\nGOLD: {}\nPRED: {}\n\n'.format(out['srce'],out['gold'],out['pred']))
                if self.args.wandb: predict_table.add_data(out['srce'],out['gold'],out['pred'])
                ofp.flush()

        scores = {}
        if self.args.evaluate:
            pred = [d['pred'] for d in outs]
            gold = [d['gold'] for d in outs]
            if 'generator' in self.args.model:
                bleu = m.evaluate.get_bleu(gold, pred)
                rouge = m.evaluate.get_rouge(gold, pred)
                cider = m.evaluate.get_cider(gold, pred)
                meteor = m.evaluate.get_meteor(gold, pred)
                scores = {'bleu':bleu, 'rouge':rouge, 'cider':cider, 'meteor':meteor}
            elif 'classifier' in self.args.model:
                scores = m.evaluate.get_cls(gold, pred)

        all_score = {f"{x}_{y}":scores[x][y] for x in scores for y in scores[x]}
        if self.args.wandb:
            wdb.log(all_score)
            wdb.log({'prediction result':predict_table})
            wdb.close()

        args_dict['run_info'] = dict()
        args_dict['run_info']['end_time'] = str(datetime.datetime.now())
        args_dict['run_info']['start_time'] = start_run_time
        args_dict['run_info']['command'] = 'python '+' '.join(sys.argv)
        ofp.write('{}\n'.format(json.dumps({'args':args_dict,'scores':scores}, ensure_ascii=False)))
        return {'args':args_dict, 'scores':scores, 'output':outs}

if __name__ == '__main__':
    logging.info('START {}'.format(start_run_time))
    logging.info('python '+' '.join(sys.argv))
    args = parse_args()

    trainer = Trainer(args)
    if args.predict:
        trainer.predict()
    else:
        trainer.train()
