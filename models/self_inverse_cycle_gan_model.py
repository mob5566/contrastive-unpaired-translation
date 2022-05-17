import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class SelfInverseCycleGANModel(BaseModel):
    """
    This class implements the self-inverse (or one-to-one) CycleGAN model, for learning image-to-image translation without paired data with one-to-one mapping constraint.

    The model training requires '--dataset_mode one2one' dataset.
    By default, the generator and discriminators are followed by CycleGAN.

    Self-inverse CycleGAN paper: https://arxiv.org/pdf/1909.04110
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(display_ncols=3)
        parser.set_defaults(gan_mode='vanilla')
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B']
        self.visual_names = ['real_A', 'fake_B', 'rec_A',
                             'real_B', 'fake_A', 'rec_B']
        if self.isTrain:
            self.model_names = ['G', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt=opt)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt=opt)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt=opt)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.real_A = input['A'].to(self.device)  # get image data A
        self.real_B = input['B'].to(self.device)  # get image data B
        self.image_paths = input['A_paths']  # get image paths

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        self.rec_A = self.netG(self.fake_B)   # G(G(A))
        self.fake_A = self.netG(self.real_B)  # G(B)
        self.rec_B = self.netG(self.fake_A)   # G(G(B))

    def forward_A(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        self.rec_A = self.netG(self.fake_B)   # G(G(A))

    def forward_B(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.fake_A = self.netG(self.real_B)  # G(B)
        self.rec_B = self.netG(self.fake_A)   # G(G(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G_A(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_A = self.opt.lambda_A

        # GAN loss D_A(G(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)

        # Forward cycle loss || G(G(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A

        # combined loss and calculate gradients
        loss_G = self.loss_G_A + self.loss_cycle_A
        loss_G.backward()

    def backward_G_B(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_B = self.opt.lambda_B

        # GAN loss D_A(G(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Forward cycle loss || G(G(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # combined loss and calculate gradients
        loss_G = self.loss_G_B + self.loss_cycle_B
        loss_G.backward()

    def data_dependent_initialize(self, data=None):
        return

    def generate_visuals_for_evaluation(self, data, mode):
        with torch.no_grad():
            visuals = {}
            AtoB = self.opt.direction == "AtoB"
            G = self.netG
            source = data["A" if AtoB else "B"].to(self.device)
            if mode == "forward":
                visuals["fake_B"] = G(source)
            else:
                raise ValueError("mode %s is not recognized" % mode)
            return visuals

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        # G_A
        self.forward_A()
        self.optimizer_G.zero_grad()  # set G_A's gradients to zero
        self.backward_G_A()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A's weights
        # G_B
        self.forward_B()
        self.optimizer_G.zero_grad()  # set G_B's gradients to zero
        self.backward_G_B()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
