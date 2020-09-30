
from model import *
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import torchvision.transforms.functional as TF
from PIL import Image


class Solver(object):
    """Solver for training and testing SPS."""
    def __init__(self, celeba_loader, celeba_loader_skt, cgui_loader_skt, rafd_loader, config, dataX, dataY):
        """Initialize configurations."""
        # Data loader.
        self.celeba_loader = celeba_loader
        self.celeba_loader_skt = celeba_loader_skt
        self.cgui_loader_skt = cgui_loader_skt
        self.rafd_loader = rafd_loader
        self.data_setX = dataX
        self.data_setY = dataY
        self.a =None

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD']:

            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            # self.G = nn.DataParallel(self.G)

            self.D = DiscriminatorY(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)
            # self.D = nn.DataParallel(self.D)

            self.G_skt = Generator2(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            # self.G_skt = nn.DataParallel(self.G_skt)

            self.D_skt = DiscriminatorY(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)
            # self.D_skt = nn.DataParallel(self.D_skt)

            self.vgg = Vgg19().cuda()
            self.vgg_loss = VGGLoss(4).cuda()
            self.style_loss = Style_Loss()

        elif self.dataset in ['Both']:
            self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector.
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.g_skt_optimizer = torch.optim.Adam(self.G_skt.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.d_skt_optimizer = torch.optim.Adam(self.D_skt.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        self.G.to(self.device)
        self.G_skt.to(self.device)
        self.D.to(self.device)
        self.D_skt.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        G_path_skt = os.path.join(self.model_save_dir, '{}-G-skt.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        D_path_skt = os.path.join(self.model_save_dir, '{}-D-skt.ckpt'.format(resume_iters))

        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.G_skt.load_state_dict(torch.load(G_path_skt, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        self.D_skt.load_state_dict(torch.load(D_path_skt, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad_g(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.g_skt_optimizer.zero_grad()


    def reset_grad_d(self):
        """Reset the gradient buffers."""

        self.d_optimizer.zero_grad()
        self.d_skt_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)



    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out


    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list


    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)

    def feature_loss(self,real, fake):
        """Compute feature loss."""
        value = 0
        for i in range(real):
            value =+ real[i] - fake[i]
        return torch.abs(value)

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
            data_loader_skt = self.celeba_loader_skt
            data_gui_loader_skt = self.cgui_loader_skt
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader


        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        data_iter_skt = iter(data_loader_skt)
        # gui_iter_skt = iter(data_gui_loader_skt)

        x_fixed, x_fixed_XY, c_org, _ = next(data_iter)
        x_fixed_skt, x_fixed_skt_XY, c_org_skt, _ = next(data_iter_skt)
        # x_gui_skt, x_gui_XY, c_org_skt, _ = next(gui_iter_skt)

        x_fixed = x_fixed.to(self.device)
        x_fixed_skt = x_fixed_skt.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
        # c_fixed_list_skt = self.create_labels(c_org_skt, self.c_dim, self.dataset, self.selected_attrs)
        c_fixed_list_skt = [c_org]


        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()

        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:

                x_real, x_real_XY, label_org, fnameX = next(data_iter)
                x_real_skt, x_real_skt_XY, label_org_skt, fnameY = next(data_iter_skt)

            except:

                data_iter = iter(data_loader)
                data_iter_skt = iter(data_loader_skt)

                x_real, x_real_XY, label_org, fnameX = next(data_iter)
                x_real_skt, x_real_skt_XY, label_org_skt, fnameY  = next(data_iter_skt)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            if self.dataset == 'CelebA':
                c_org = label_org.clone()
                c_trg = label_trg.clone()

            elif self.dataset == 'RaFD':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)

            # x_real = x_real[:,:3,:].to(self.device) # Input images.
            x_real = x_real.to(self.device)           # Input images.
            x_real_skt = x_real_skt.to(self.device)           # Input images.
            x_real_skt_XY = x_real_skt_XY.to(self.device)           # Input images.
            x_real_XY = x_real_XY.to(self.device)           # Input images.

            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.

            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_org_skt = label_org_skt.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.
            # c_fixed_list_skt = c_fixed_list_skt[0].to(self.device)

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            # print(x_real[:,:3,:].shape)
            out_src, out_cls = self.D(x_real_skt_XY)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset) * 0.1

            # Compute loss with fake images.
            x_fake = self.G_skt(x_real_skt, c_trg)
            out_src, out_cls = self.D_skt(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real_skt.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real_skt.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D_skt(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat) * 10


            # Compute loss with real sketches
            out_src, out_cls = self.D_skt(x_real_skt)
            d_loss_real_skt = - torch.mean(out_src)

            x_fake = self.G(x_real_skt, c_trg)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake_skt = torch.mean(out_src)
            d_loss_cls_skt = self.classification_loss(out_cls, label_trg, self.dataset) * 0.1


            # Compute loss for gradient penalty sketch.
            alpha = torch.rand(x_real_skt_XY.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real_skt_XY.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp_skt = self.gradient_penalty(out_src, x_hat) * 10


            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp \
                    + (d_loss_real_skt + d_loss_fake_skt + d_loss_gp_skt + d_loss_cls_skt)
            self.reset_grad_d()
            d_loss.backward()
            self.d_optimizer.step()
            self.d_skt_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/d_loss_cls_skt'] = d_loss_cls_skt.item()
            loss['D/loss_real_skt'] = d_loss_real_skt.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_fake_skt'] = d_loss_fake_skt.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            loss['D/loss_gp_skt'] = d_loss_gp_skt.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if i > 0:
            # if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real_skt, label_org_skt)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_recXY = 0

                    # torch.mean(torch.abs(x_real_skt_XY - x_fake))
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset) * 0.1
                g_loss_fakeXY = 0


                # Target-to-original domain.
                x_reconst = self.G_skt(x_fake, c_org)
                out_src, _ = self.D_skt(x_reconst)
                g_loss_fake_rec = - torch.mean(out_src)

                g_loss_rec = torch.mean(torch.abs(x_real_skt - x_reconst)) *  self.lambda_rec
                B_vgg, B_rec_vgg = self.vgg(x_real_skt, label_org), self.vgg(x_reconst, label_org)
                styl_loss_0 =  self.style_loss(B_vgg, B_rec_vgg) * 100
                # exit()


                x_fake = self.G_skt(x_real_skt_XY, label_org_skt)
                out_src, _ = self.D_skt(x_fake)
                g_loss_fake_skt = - torch.mean(out_src)

                # Target-to-original domain.
                x_reconst = self.G(x_fake, label_org_skt)
                A_vgg, A_rec_vgg = self.vgg(x_real_skt_XY, label_org), self.vgg(x_reconst, label_org)
                styl_loss_1 = self.style_loss(A_vgg, A_rec_vgg) * 100
                loss_av = 0
                loss_ar = 0

                g_loss_recXY = torch.mean(torch.abs(x_real_skt_XY - x_reconst)) * self.lambda_rec
                g_loss_cls_vgg = loss_av + loss_ar
                g_loss_rec_img  = 0
                # g_loss_rec_img  = self.vgg_loss(A_vgg, A_rec_vgg)

                # Backward and optimize.
                g_loss = g_loss_fakeXY + g_loss_recXY + g_loss_fake +g_loss_rec + self.lambda_cls * g_loss_cls + styl_loss_0 + styl_loss_1+ g_loss_cls_vgg *  self.lambda_cls + \
                         g_loss_fake + g_loss_fake_rec + g_loss_fake_skt + g_loss_rec_img

                self.reset_grad_g()
                g_loss.backward()
                self.g_optimizer.step()
                self.g_skt_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/g_loss_recXY'] = g_loss_recXY
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    x_fixed_skt_XY = x_fixed_skt_XY.to(device)
                    x_fixed_skt = x_fixed_skt.to(device)
                    x_skt_listXY = [x_fixed_skt_XY]
                    x_fake_list = [x_fixed_skt]
                    # for c_fixed in c_fixed_list_skt:
                    c_fixed = c_fixed_list_skt[0]
                    c_fixed = c_fixed.to(device)
                    x_fake_list.append(self.G(x_fixed_skt, c_fixed))
                        # x_skt_listXY.append(self.G(x_fixed_skt_XY, c_fixed))
                    x_skt_cat = (x_skt_listXY + x_fake_list)
                    x_concat = torch.cat(x_skt_cat, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                G_path_skt = os.path.join(self.model_save_dir, '{}-G-skt.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                D_path_skt = os.path.join(self.model_save_dir, '{}-D-skt.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.G_skt.state_dict(), G_path_skt)
                torch.save(self.D.state_dict(), D_path)
                torch.save(self.D_skt.state_dict(), D_path_skt)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
            data_loader_skt = self.celeba_loader_skt
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # self.G.eval()
        with torch.no_grad():
            for i, (x_real,X ,c_org,fn) in enumerate(data_loader_skt):
                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                X = X.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                # Translate images.
                x_fake_list = [X,x_real]
                # for c_trg in c_trg_list[0]:
                c_trg = c_trg_list[0]
                # print(c_trg)
                f_img = (self.G(x_real, c_trg))
                x_fake_list.append(f_img)
                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                # print(i)
                # print(fn[0])
                # import  matplotlib.pyplot as plt
                # plt.subplot(1,3,1)
                # plt.imshow(f_img.squeeze(0).permute(1,2,0).cpu())
                # plt.subplot(1, 3, 2)
                # plt.imshow(X.squeeze(0).permute(1, 2, 0).cpu())
                # plt.subplot(1, 3, 3)
                # plt.imshow(x_real.squeeze(0).permute(1, 2, 0).cpu()) #Sketch
                # plt.show()
                # exit()
                # result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                result_path1 = os.path.join(self.result_dir, 'isketch_' + fn[0] )
                result_path2 = os.path.join(self.result_dir, 'image_' + fn[0]  )
                save_image(self.denorm(x_real.data.cpu()), result_path1, nrow=1, padding=0)
                save_image(self.denorm(f_img.data.cpu()), result_path2, nrow=1, padding=0)
                # print('Saved real and fake images into {}...'.format(result_path))


    def test_gui(self):
        """Translate images using GUI trained on a single dataset."""
        # Load the trained generator.
        print("GUI Test in Session")
        self.restore_model(self.test_iters)

        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
            # data_loader_skt = self.celeba_loader_skt
            data_loader_skt = self.cgui_loader_skt
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader



        with torch.no_grad():
            for i, (x_real,_,_) in enumerate(data_loader_skt):
                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                # X = X.to(self.device)
                c_org = [[1., 0., 1., 0., 1.]]
                c_org = torch.Tensor(c_org)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list[:4]:
                    # print(c_trg)
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.png'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                image_file = '{}'.format(result_path)
                # print('Saved real and fake images into {}...'.format(result_path))
                # X = X.squeeze(0).permute(1, 2, 0).cpu()
                yield image_file

