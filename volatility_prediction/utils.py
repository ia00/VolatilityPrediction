import einops
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

n_features = 21
n_stocks = 112
n_seconds = 600


def prepare_data(stock_id, stock_ind, set, time_ids, coarsen, norm, out):
    df_book = pd.read_parquet(
        f"data/optiver-realized-volatility-prediction/book_{set}.parquet/stock_id={stock_id}"
    )
    df_min_second = df_book.groupby("time_id").agg(
        min_second=("seconds_in_bucket", "min")
    )
    df_book = (
        df_book.merge(df_min_second, left_on="time_id", right_index=True)
        .eval("seconds_in_bucket = seconds_in_bucket - min_second")
        .drop("min_second", axis=1)
    )
    df_trade = (
        pd.read_parquet(
            f"data/optiver-realized-volatility-prediction/trade_{set}.parquet/stock_id={stock_id}"
        )
        .merge(df_min_second, left_on="time_id", right_index=True)
        .eval("seconds_in_bucket = seconds_in_bucket - min_second")
        .drop("min_second", axis=1)
    )
    df = pd.merge(df_book, df_trade, on=["time_id", "seconds_in_bucket"], how="outer")
    df["stock_id"] = stock_id
    df = df.set_index(["stock_id", "time_id", "seconds_in_bucket"])
    df = df.to_xarray().astype("float32")
    df = df.reindex({"time_id": time_ids, "seconds_in_bucket": np.arange(n_seconds)})
    for name in [
        "bid_price1",
        "bid_price2",
        "ask_price1",
        "ask_price2",
        "bid_size1",
        "bid_size2",
        "ask_size1",
        "ask_size2",
    ]:
        df[name] = df[name].ffill("seconds_in_bucket")
    df["wap1"] = (df.bid_price1 * df.ask_size1 + df.ask_price1 * df.bid_size1) / (
        df.bid_size1 + df.ask_size1
    )
    df["wap2"] = (df.bid_price2 * df.ask_size2 + df.ask_price2 * df.bid_size2) / (
        df.bid_size2 + df.ask_size2
    )
    df["log_return1"] = np.log(df.wap1).diff("seconds_in_bucket")
    df["log_return2"] = np.log(df.wap2).diff("seconds_in_bucket")
    df["current_vol"] = (df.log_return1**2).sum("seconds_in_bucket") ** 0.5
    df["current_vol_2nd_half"] = (df.log_return1[..., 300:] ** 2).sum(
        "seconds_in_bucket"
    ) ** 0.5
    if coarsen > 1:
        mean_features = [
            "ask_price1",
            "ask_price2",
            "bid_price1",
            "bid_price2",
            "ask_size1",
            "ask_size2",
            "bid_size1",
            "bid_size2",
            "price",
        ]
        sum_features = ["size", "order_count"]

        df = xr.merge(
            (
                df[mean_features]
                .coarsen({"seconds_in_bucket": coarsen}, coord_func="min")
                .mean(),
                df[sum_features]
                .coarsen({"seconds_in_bucket": coarsen}, coord_func="min")
                .sum(),
                df[["current_vol", "current_vol_2nd_half"]],
            )
        )
        df["wap1"] = (df.bid_price1 * df.ask_size1 + df.ask_price1 * df.bid_size1) / (
            df.bid_size1 + df.ask_size1
        )
        df["wap2"] = (df.bid_price2 * df.ask_size2 + df.ask_price2 * df.bid_size2) / (
            df.bid_size2 + df.ask_size2
        )
        df["log_return1"] = np.log(df.wap1).diff("seconds_in_bucket")
        df["log_return2"] = np.log(df.wap2).diff("seconds_in_bucket")

    df["spread1"] = df.ask_price1 - df.bid_price1
    df["spread2"] = df.ask_price2 - df.ask_price1
    df["spread3"] = df.bid_price1 - df.bid_price2
    df["total_volume"] = df.ask_size1 + df.ask_size2 + df.bid_size1 + df.bid_size2
    df["volume_imbalance1"] = df.ask_size1 + df.ask_size2 - df.bid_size1 - df.bid_size2
    df["volume_imbalance2"] = (
        df.ask_size1 + df.ask_size2 - df.bid_size1 - df.bid_size2
    ) / df.total_volume
    for name in [
        "bid_size1",
        "bid_size2",
        "ask_size1",
        "ask_size2",
        "size",
        "order_count",
        "total_volume",
    ]:
        df[name] = np.log1p(df[name])
    #         df[name] = df[name].rank('seconds_in_bucket')
    df["volume_imbalance1"] = np.sign(df["volume_imbalance1"]) * np.log1p(
        abs(df["volume_imbalance1"])
    )

    df = df.fillna(
        {
            "ask_price1": 1,
            "ask_price2": 1,
            "bid_price1": 1,
            "bid_price2": 1,
            "ask_size1": 0,
            "ask_size2": 0,
            "bid_size1": 0,
            "bid_size2": 0,
            "price": 1,
            "size": 0,
            "order_count": 0,
            "wap1": 1,
            "wap2": 1,
            "log_return1": 0,
            "log_return2": 0,
            "spread1": 0,
            "spread2": 0,
            "spread3": 0,
            "total_volume": 0,
            "volume_imbalance1": 0,
            "volume_imbalance2": 0,
            "current_vol": 0,
            "current_vol_2nd_half": 0,
        }
    )
    features = [
        "ask_price1",
        "ask_price2",
        "bid_price1",
        "bid_price2",
        "ask_size1",
        "ask_size2",
        "bid_size1",
        "bid_size2",
        "price",
        "size",
        "order_count",
        "wap1",
        "wap2",
        "log_return1",
        "log_return2",
        "spread1",
        "spread2",
        "spread3",
        "total_volume",
        "volume_imbalance1",
        "volume_imbalance2",
    ]
    extra = ["current_vol", "current_vol_2nd_half"]

    if norm is not None:
        mean = norm["mean"].sel(stock_id=stock_id)
        std = norm["std"].sel(stock_id=stock_id)
    else:
        mean = df.mean(("time_id", "seconds_in_bucket")).drop(
            ["current_vol", "current_vol_2nd_half"]
        )
        std = df.std(("time_id", "seconds_in_bucket")).drop(
            ["current_vol", "current_vol_2nd_half"]
        )

    df.update((df - mean) / std)
    df = df.astype("float32")

    out[:, stock_ind] = einops.rearrange(
        df[features].to_array().values, "f () t sec -> t sec f"
    )
    return df[extra], {"mean": mean, "std": std}


def load_and_prepare_data(data_dir, coarsen=3, n_jobs=4):
    df_train = pd.read_csv(f"{data_dir}/train.csv")
    train_data = np.memmap(
        "/tmp/train.npy",
        "float16",
        "w+",
        shape=(df_train.time_id.nunique(), n_stocks, n_seconds // coarsen, n_features),
    )

    res = Parallel(n_jobs=n_jobs, verbose=51)(
        delayed(prepare_data)(
            stock_id,
            stock_ind,
            "train",
            df_train.time_id.unique(),
            coarsen,
            None,
            train_data,
        )
        for stock_ind, stock_id in enumerate(df_train.stock_id.unique())
    )

    train_extra = xr.concat([x[0] for x in res], "stock_id")
    train_extra["target"] = (
        df_train.set_index(["time_id", "stock_id"])
        .to_xarray()["target"]
        .astype("float32")
    )
    train_extra = train_extra.transpose("time_id", "stock_id")

    train_norm = {
        "mean": xr.concat([x[1]["mean"] for x in res], "stock_id"),
        "std": xr.concat([x[1]["std"] for x in res], "stock_id"),
    }

    return np.array(train_data), train_extra, train_norm
