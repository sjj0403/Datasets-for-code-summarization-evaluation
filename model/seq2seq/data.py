from torch.utils.data import Dataset

import utils


class CodePtrDataset(Dataset):

    def __init__(self, code_path, nl_path):
        # get lines
        codes = utils.load_dataset(code_path)
        nls = utils.load_dataset(nl_path)

        if len(codes) != len(nls):
            raise Exception('The lengths of three dataset do not match.')

        self.codes, self.nls = utils.filter_data(codes, nls)

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, index):
        return self.codes[index], self.nls[index]

    def get_dataset(self):
        return self.codes, self.nls
