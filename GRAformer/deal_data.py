from data_provider.data_factory import data_provider



class Exp_Main():

    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

if __name__ == '__main__':
    exp=Exp_Main()

    train_data, train_loader = exp._get_data(flag='train')
    vali_data, vali_loader = exp._get_data(flag='val')
    test_data, test_loader = exp._get_data(flag='test')