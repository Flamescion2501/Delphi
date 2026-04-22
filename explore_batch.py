import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import polars as pl
    import torch

    from utils import get_batch, get_p2i

    return Path, get_batch, get_p2i, np, pl, torch


@app.cell
def _(Path, np):
    data_dir = Path("data/ukb_simulated_data/")
    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    return train_data, val_data


@app.cell
def _(get_p2i, train_data, val_data):
    train_p2i = get_p2i(train_data)
    val_p2i = get_p2i(val_data)
    return (train_p2i,)


@app.cell
def _(torch, train_data, train_p2i):
    batch_size = 128
    data = train_data
    p2i = train_p2i
    ix = torch.randint(len(p2i), (batch_size,))
    return data, ix, p2i


@app.cell
def _(ix):
    ix.shape
    return


@app.cell
def _(data, get_batch, ix, p2i):
    X, A, Y, B = get_batch(
        ix, data, p2i, block_size=48, device="cpu", select="left", no_event_token_rate=5, cut_batch=True
    )
    return X, Y


@app.cell
def _(X):
    X.shape
    return


@app.cell
def _(ix, p2i, pl):
    _df = pl.from_numpy(p2i).with_row_index().filter(pl.col("index") == ix[0])
    i = _df.item(0, 1)
    _df
    return (i,)


@app.cell
def _(X, Y):
    print(X[0])
    print(Y[0])
    return


@app.cell
def _(i, pl, train_data):
    _df = pl.from_numpy(train_data).with_row_index().filter(pl.col("index") >= i)
    id = _df.item(0, 1)
    _df.filter(pl.nth(1) == id)
    return


if __name__ == "__main__":
    app.run()
