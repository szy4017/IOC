class Config(object):
    def __init__(self):
        # datasets
        self.dataset = 'RTF'
        self.img_size = (224, 224)
        # model configs
        self.net_name = 'hsr_res18'  # ['hsr_LeNet', 'hsr_res18','res18_vae']
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
        self.batch_size = 4
        self.weight_decay = 1e-6



