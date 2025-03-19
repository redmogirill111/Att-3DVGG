class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = r'F:\dataset\huoyanshujvji11111111\20220313\2-4\YuanMP4'

            # Save preprocess data into output_dir
            # output_dir = r'F:\dataset\huoyanshujvji11111111\20220313\2-4\RongHe'
            output_dir = r"Y:\dateset\4Chanel"

            return root_dir, output_dir
        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = '/Path/to/hmdb-51'

            output_dir = '/path/to/VAR/hmdb51'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return r'data/c3d-pretrained.pth'
