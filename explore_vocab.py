import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import altair as alt
    import polars as pl
    import torch
    from torch.nn import functional

    from model import DelphiConfig
    from model_mod import DelphiMod

    return DelphiConfig, DelphiMod, alt, functional, pl, torch


@app.cell
def _(torch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return (device,)


@app.cell
def _(pl):
    labels_df = pl.read_csv("delphi_labels_chapters_colours_icd.csv", columns=["index", "name"])
    labels_df
    return (labels_df,)


@app.cell
def _(pl):
    _sample_trajectory = [
        ("Male", 0),
        ("B01 Varicella [chickenpox]", 2),
        ("L20 Atopic dermatitis", 3),
        ("No event", 5),
        ("No event", 10),
        ("No event", 15),
        ("No event", 20),
        ("G43 Migraine", 20),
        ("E73 Lactose intolerance", 21),
        ("B27 Infectious mononucleosis", 22),
        ("No event", 25),
        ("J11 Influenza, virus not identified", 28),
        ("No event", 30),
        ("No event", 35),
        ("No event", 40),
        ("Smoking low", 41),
        ("BMI mid", 41),
        ("Alcohol low", 41),
        ("No event", 42),
    ]
    sample_df = pl.DataFrame(
        {
            "age": [item[1] for item in _sample_trajectory],
            "token": [item[0] for item in _sample_trajectory],
        }
    ).with_columns(pl.col.age.cast(pl.Float32))
    sample_df
    return (sample_df,)


@app.cell
def _(labels_df, pl, sample_df):
    event_input = (
        sample_df["token"]
        .replace(labels_df["name"], labels_df["index"])
        .cast(pl.Int64)
        .to_torch()
        .unsqueeze(0)
    )
    return (event_input,)


@app.cell
def _(event_input):
    event_input
    return


@app.cell
def _(sample_df):
    age_input = (sample_df["age"].to_torch() * 365.25).unsqueeze(0)
    return (age_input,)


@app.cell
def _(torch):
    checkpoint = torch.load("Delphi-2M/ckpt.pt")
    return (checkpoint,)


@app.cell
def _(DelphiConfig, DelphiMod, checkpoint, device):
    model = DelphiMod(DelphiConfig(**checkpoint["model_args"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(device)
    return (model,)


@app.cell
def _(age_output, event_output, logits, t):
    event_output.shape, age_output.shape, logits.shape, t.shape
    return


@app.cell
def _(age_input, device, event_input, functional, model, sample_df, torch):
    torch.manual_seed(2501)
    with torch.device(device):
        event_output, age_output, logits, t = model.generate(
            event_input.to(device),
            age_input.to(device),
            max_new_tokens=100,
            termination_tokens=[1269],
        )

    age_prediction = age_output.flatten()[len(sample_df) :] / 365.25
    age_prediction = [round(age, ndigits=1) for age in age_prediction.tolist()]

    event_prediction = event_output.flatten()[len(sample_df) :].tolist()

    probs = functional.softmax(logits[0, len(sample_df) - 1 : -1, :], dim=1)

    logits_top_k = probs.topk(20, dim=1)
    logtis_top_k_list = []
    for _es, _ps in zip(logits_top_k.indices, logits_top_k.values):
        logtis_top_k_list.append([])
        for _e, _p in zip(_es, _ps):
            logtis_top_k_list[-1].append({"event": _e, "prob": round(_p.item(), ndigits=4)})

    logits_min_t_list = []
    for _e, _p in zip(event_prediction, probs):
        logits_min_t_list.append([{"event": _e, "prob": round(_p[_e].item(), ndigits=4)}])

    t = t[0]
    t_top_k = t.topk(k=20, dim=1, largest=False)
    t_top_k_list = []
    for _es, _ys in zip(t_top_k.indices, t_top_k.values):
        t_top_k_list.append([])
        for _e, _y in zip(_es, _ys):
            t_top_k_list[-1].append({"event": _e, "years": round((_y / 365.25).item(), ndigits=1)})
    return (
        age_output,
        age_prediction,
        event_output,
        event_prediction,
        logits,
        logits_min_t_list,
        logtis_top_k_list,
        probs,
        t,
        t_top_k_list,
    )


@app.cell
def _(
    age_prediction,
    event_prediction,
    labels_df,
    logits_min_t_list,
    logtis_top_k_list,
    mo,
    pl,
    t_top_k_list,
):
    prediction_df = pl.DataFrame(
        {
            "age": age_prediction,
            "event": event_prediction,
            "logits_top_k": logtis_top_k_list,
            "logits_min_t": logits_min_t_list,
            "t_top_k": t_top_k_list,
        }
    ).with_columns(
        pl.col.event.cast(pl.String).replace(labels_df["index"], labels_df["name"]),
    )
    prediction_tb = mo.ui.table(
        prediction_df,
        selection="single",
        pagination=False,
        show_column_summaries=False,
        show_data_types=False,
    )
    prediction_tb
    return (prediction_tb,)


@app.cell
def _(labels_df, pl):
    def expand_column(df: pl.DataFrame, column_name: str) -> pl.DataFrame:
        return (
            df.explode(column_name)
            .unnest(column_name)
            .with_columns(
                pl.col("event").cast(pl.String).replace(labels_df["index"], labels_df["name"])
            )
        )

    return (expand_column,)


@app.cell
def _(expand_column, mo, prediction_tb):
    mo.stop(prediction_tb.value.is_empty())

    logits_top_k_df = prediction_tb.value.select("logits_top_k").pipe(expand_column, "logits_top_k")
    logits_min_t_df = prediction_tb.value.select("logits_min_t").pipe(expand_column, "logits_min_t")
    t_top_k_df = prediction_tb.value.select("t_top_k").pipe(expand_column, "t_top_k")
    return logits_min_t_df, logits_top_k_df, t_top_k_df


@app.cell
def _(alt, logits_min_t_df, logits_top_k_df, mo, t_top_k_df):
    logits_top_k_graph = (
        alt.Chart(logits_top_k_df).mark_bar(tooltip=True).encode(alt.X("event", sort=None), y="prob")
    )

    xrule = alt.Chart().mark_rule(color="red").encode(x=alt.datum(logits_min_t_df["event"].item()))

    yrule = alt.Chart().mark_rule(color="red").encode(y=alt.datum(logits_min_t_df["prob"].item()))

    t_top_k_graph = (
        alt.Chart(t_top_k_df).mark_bar(tooltip=True).encode(alt.X("event", sort=None), y="years")
    )

    mo.hstack([logits_top_k_graph + xrule + yrule, t_top_k_graph])
    return


@app.cell
def _(pl, torch):
    missing = torch.tensor(
        pl.read_csv("../DelphiAnalysis/missing.csv")
        .filter(pl.col.label.str.contains(r"^\w{3} "))["index"]
        .to_list()
    )
    missing, missing.shape
    return (missing,)


@app.cell
def _(missing, probs, torch):
    mask = torch.ones_like(probs[0, :], dtype=torch.bool)
    mask[missing] = False
    mask, mask.sum()
    return (mask,)


@app.cell
def _(mask, probs, torch):
    probs_missing = torch.detach_copy(probs)
    probs_missing[:, mask] = float("-inf")
    probs_missing, probs_missing.shape
    return (probs_missing,)


@app.cell
def _(probs, probs_missing, t):
    for _i in range(probs.shape[0]):
        _sorted_t_values, _sorted_t_indices = t[_i].sort()
        _max_missing_index_in_probs = (
            probs[_i].sort(descending=True).indices == probs_missing[_i].argmax()
        ).nonzero(as_tuple=True)[0]
        _max_missing_index_in_t = (_sorted_t_indices == probs_missing[_i].argmax()).nonzero(
            as_tuple=True
        )[0]
        print(
            "Missing token with highest prob: token_id="
            + str(probs_missing[_i].argmax().item())
            + ", prob="
            + "{:.3e}".format(probs_missing[_i].max().item())
        )
        print("Token overall ranking: " + str(_max_missing_index_in_probs.item()))
        print(
            "Time to token: ranking="
            + str(_max_missing_index_in_t.item())
            + ", time="
            + str(_sorted_t_values[_max_missing_index_in_t].item())
        )
        print()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
