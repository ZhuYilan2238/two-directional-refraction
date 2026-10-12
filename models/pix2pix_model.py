import torch
from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_HV', type=float, default=10.0,
                                help='weight for physics consistency loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_GAN', 'G_L1', 'G_HV', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        self.fake_other = None

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)  
        self.real_B = input['B' if AtoB else 'A'].to(self.device)  
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    @staticmethod
    def diff_x(t: torch.Tensor) -> torch.Tensor:
        return t[:, :, :, 1:] - t[:, :, :, :-1]

    @staticmethod
    def diff_y(t: torch.Tensor) -> torch.Tensor:
        return t[:, :, 1:, :] - t[:, :, :-1, :]

    def get_direction(self) -> torch.Tensor:
        ind = self.real_A[:, -1:, :, :]
        cond = ind.mean(dim=(2, 3), keepdim=True)  
        vertical = (cond > 0).float()
        return vertical

    def forward(self):
        self.fake_B = self.netG(self.real_A)  
        self.fake_other = None
        if self.isTrain and getattr(self.opt, 'lambda_HV', 0.0) > 0:
            real_A_flip = self.real_A.clone()
            real_A_flip[:, -1:, :, :] = -real_A_flip[:, -1:, :, :]
            self.fake_other = self.netG(real_A_flip)

    def backward_D(self, backward=True):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        if backward:
            self.loss_D.backward()

    def backward_G(self, backward=True):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G_HV = 0.0
        if getattr(self.opt, 'lambda_HV', 0.0) > 0 and self.fake_other is not None:
            vertical = self.get_direction()  
            gx = vertical * self.fake_other + (1.0 - vertical) * self.fake_B
            gy = vertical * self.fake_B     + (1.0 - vertical) * self.fake_other
            dy_gx = self.diff_y(gx)  
            dx_gy = self.diff_x(gy)  
            dy_gx = dy_gx[:, :, :, :-1]
            dx_gy = dx_gy[:, :, :-1, :]
            curl = dy_gx - dx_gy
            self.loss_G_HV = curl.abs().mean() * self.opt.lambda_HV
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_HV
        if backward:
            self.loss_G.backward()

    def val_loss(self):
        with torch.no_grad():
            self.forward()
            self.backward_D(backward=False)
            self.backward_G(backward=False)

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
