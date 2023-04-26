from .BaseDataLoaderWrapper import BaseDataLoaderWrapper


class XyDataLoaderWrapper(BaseDataLoaderWrapper):
    def split_iter_data(self, iter_data):
        self.iter_data['x'], self.iter_data['y'] = iter_data
