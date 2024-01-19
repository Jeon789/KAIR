from collections import OrderedDict
import math
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam

from models.select_network import define_G
from models.model_base import ModelBase
from models.loss import CharbonnierLoss
from models.loss_ssim import SSIMLoss

from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip


class ModelPlain(ModelBase):
    """Train with pixel loss"""
    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'], param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_lossfn_type == 'ssim':
            self.G_lossfn = SSIMLoss().to(self.device)
        elif G_lossfn_type == 'charbonnier':
            self.G_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']

        self.G_loss_form = self.opt_train['G_loss_form']

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        if self.opt_train['G_optimizer_type'] == 'adam':
            self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'],
                                    betas=self.opt_train['G_optimizer_betas'],
                                    weight_decay=self.opt_train['G_optimizer_wd'])
        else:
            raise NotImplementedError

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        if self.opt_train['G_scheduler_type'] == 'MultiStepLR':
            self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                            self.opt_train['G_scheduler_milestones'],
                                                            self.opt_train['G_scheduler_gamma']
                                                            ))
        elif self.opt_train['G_scheduler_type'] == 'CosineAnnealingWarmRestarts':
            self.schedulers.append(lr_scheduler.CosineAnnealingWarmRestarts(self.G_optimizer,
                                                            self.opt_train['G_scheduler_periods'],
                                                            self.opt_train['G_scheduler_restart_weights'],
                                                            self.opt_train['G_scheduler_eta_min']
                                                            ))
        else:
            raise NotImplementedError

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        if need_H:
            self.H = data['H'].to(self.device)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        # self.E = self.netG(self.L)

        # x=H / y=L / E=f(y) / P=f(x)

        # loss_form 0 : basic : x-f(y)
        if self.G_loss_form == 0 :
            self.E = self.netG(self.L)

        # loss_form 1 : add x-f(x)
        if self.G_loss_form == 1 :
            self.E = self.netG(self.L)
            self.P = self.netG(self.H)

        # loss_form 2 : add f(x)-f(y)
        if self.G_loss_form == 2 :
            self.E = self.netG(self.L)
            self.P = self.netG(self.H)


        # loss_form 3 : add f(x)-f2(x)
        if self.G_loss_form == 3 :
            self.E = self.netG(self.L)
            self.P  = self.netG(self.H)
            self.PP = self.netG(self.netG(self.H))


        # loss_form 4 : add x-f(x), f(x)-f(y)
        if self.G_loss_form == 4 :
            self.E = self.netG(self.L)
            self.P = self.netG(self.H)


        # loss_form 5 : heron
        if self.G_loss_form == 5 or self.G_loss_form == 6 :
            self.E = self.netG(self.L)
            self.P = self.netG(self.H)



    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward()

        # G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)

        # -----------------------------------custom---------------------------------------
        G_loss = 0
        if self.G_loss_form == 0 :
            G_loss += self.G_lossfn_weight * self.G_lossfn(self.E, self.H)

        if self.G_loss_form == 1 :
            G_loss += self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
            G_loss += self.G_lossfn_weight * self.G_lossfn(self.H, self.P)

        if self.G_loss_form == 2 :
            G_loss += self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
            G_loss += self.G_lossfn_weight * self.G_lossfn(self.P, self.E)

        if self.G_loss_form == 3 :
            G_loss += self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
            G_loss += self.G_lossfn_weight * self.G_lossfn(self.P, self.PP)

        if self.G_loss_form == 4 :
            G_loss += self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
            G_loss += self.G_lossfn_weight * self.G_lossfn(self.H, self.P)
            G_loss += self.G_lossfn_weight * self.G_lossfn(self.P, self.E)

        if self.G_loss_form == 5 :
            pdist = torch.nn.PairwiseDistance(p=2)
            batch_size = self.E.size()[0]
            x = self.E.view(batch_size, -1)
            f_x = self.P.view(batch_size, -1)
            f_y = self.E.view(batch_size, -1)
            a, b, c = pdist(x, f_x), pdist(x, f_y), pdist(f_x, f_y)
            s = (a+b+c)/2
            area = torch.sqrt( s*(s-a)*(s-b)*(s-c) )
            G_loss += self.G_lossfn_weight * ( torch.mean(area) / torch.sqrt(x.size()[1]) )


        if self.G_loss_form == 6 :
            pdist = torch.nn.PairwiseDistance(p=2)
            batch_size = self.E.size()[0]
            x = self.H.view(batch_size, -1)
            f_x = self.P.view(batch_size, -1)
            f_y = self.E.view(batch_size, -1)
            a, b, c = pdist(x, f_x), pdist(x, f_y), pdist(f_x, f_y)
            s = (a+b+c)/2
            area = torch.sqrt( s*(s-a)*(s-b)*(s-c) )
            G_loss += self.G_lossfn_weight * ( torch.mean(area) / math.sqrt(x.size()[1]) )

            # 세변의 길이를 비슷하게 만든다. 정삼각형처럼 만든다.
            # 삼각형이 무너져버리는 경우를 방지하기 위하여.
            abc = torch.stack([a,b,c],dim=1)
            regularizer = self.G_lossfn(torch.max(abc,1).values, torch.min(abc,1).values)
            G_loss += 0.1 * self.G_lossfn_weight * regularizer
        # -----------------------------------custom---------------------------------------


        G_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['G_loss'] = G_loss.item()

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()

    # ----------------------------------------
    # test / inference x8
    # ----------------------------------------
    def testx8(self):
        self.netG.eval()
        with torch.no_grad():
            self.E = test_mode(self.netG, self.L, mode=3, sf=self.opt['scale'], modulo=1)
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
