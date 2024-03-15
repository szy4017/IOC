class Config(object):
    def __init__(self):
        # datasets
        self.dataset = 'OCL'
        # model configs
        self.d_model = 64
        self.d_res = 32
        self.num_inv = 4
        self.link = 'densenet'
        self.res_con = 0
        self.clamp = 0.2

        # train configs
        self.optimizer = 'adam'
        self.lr = 0.001      
        self.num_epoch = 100
        self.change_center_epoch = 10
        self.lr_milestones = [50,100]
        self.batch_size = 4
        self.weight_decay = 1e-6