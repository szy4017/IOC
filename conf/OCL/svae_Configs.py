class Config(object):
    def __init__(self):
        # datasets
        self.dataset = 'OCL'
        self.img_size = (320, 1440)
        # model configs
        self.project_channels = 512
        self.center_eps = 0.1
        self.objective = 'one-class'  #['one-class', 'soft-boundary']
        self.nu = 0.1    # must be 0 < nu <= 1

        # train configs
        self.optimizer = 'adam'
        self.lr = 0.001      
        self.num_epoch = 100
        self.change_center_epoch = 10
        self.lr_milestones = [50,100]
        self.batch_size = 8
        self.weight_decay = 1e-6



