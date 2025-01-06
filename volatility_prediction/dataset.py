from torch.utils.data import Dataset


class OptiverDataset(Dataset):
    def __init__(self, features_data, extra_data, mode, time_ids, n_stocks):
        self.features_data = features_data
        self.extra_data = extra_data
        self.time_ids = time_ids
        self.mode = mode
        self.n_stocks = n_stocks

    def __len__(self):
        if self.mode == "single-stock":
            return len(self.time_ids) * self.n_stocks
        elif self.mode == "multi-stock":
            return len(self.time_ids)

    def __getitem__(self, i):
        if self.mode == "single-stock":
            time_id = self.time_ids[i // self.n_stocks]
            time_ind = self.extra_data.indexes["time_id"].get_loc(time_id)
            stock_ind = i % self.n_stocks
            stock_id = self.extra_data.indexes["stock_id"][stock_ind]
            return {
                "data": self.features_data[time_ind],
                "target": self.extra_data["target"].values[time_ind, stock_ind],
                "current_vol": self.extra_data["current_vol"].values[
                    time_ind, stock_ind
                ],
                "current_vol_2nd_half": self.extra_data["current_vol_2nd_half"].values[
                    time_ind, stock_ind
                ],
                "time_id": time_id,
                "stock_id": stock_id,
                "stock_ind": stock_ind,
            }
        elif self.mode == "multi-stock":
            time_id = self.time_ids[i]
            time_ind = self.extra_data.indexes["time_id"].get_loc(time_id)
            return {
                "data": self.features_data[time_ind],
                "target": self.extra_data["target"].values[time_ind],
                "current_vol": self.extra_data["current_vol"].values[time_ind],
                "current_vol_2nd_half": self.extra_data["current_vol_2nd_half"].values[
                    time_ind
                ],
                "time_id": time_id,
            }
