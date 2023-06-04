

class ParametrizedBatchProducer:

    def __init__(self, df, batch_size, shuffle=True):
        """A class used to select what data to produce when a batch of data is requested.

        Args:
            df (pandas DataFrame): dataframe which the producer uses. Usually a subset of all samples with the same class.
            batch_size (int): batch size to produce
            shuffle (bool, optional): whether to shuffle the DF after all samples were produced. Defaults to True.
        """
        self.df = df.sample(frac=1)
        self.eval_df = None
        self.idx = 0
        self.batch_size = batch_size
        self.len = len(self.df)
        self.shuffle = shuffle

    def __len__(self):
        return len(self.df)

    def produce_batch(self, eval_data=False):
        """Produces a batch (in the form of a pandas DataFrame) from its available samples.

        Args:
            eval_data (bool, optional): if a batch produced is eval_data, these samples are stored separately and not generated as training samples. Defaults to False.

        Returns:
            pandas DataFrame: dataframe of selected samples
        """
        select_df = self.df[self.idx *
                            self.batch_size: (self.idx + 1) * self.batch_size]
        self.idx += 1
        if self.shuffle == True and self.idx * self.batch_size > self.len:
            self.idx = 0
            self.df = self.df.sample(frac=1)
        # if batch is smaller in size than it is supposed to be, append the rest
        if len(select_df) < self.batch_size:
            select_df = select_df.append(
                self.df[0:self.batch_size - len(select_df)])

        # If generating eval data, drop the records from train data.
        if eval_data == True:
            self.df = self.df.drop(select_df.index)
            # store the records as eval data. Just in case.
            if self.eval_df is None:
                self.eval_df = select_df
            else:
                self.eval_data = self.eval_df.append(select_df)

        return select_df
