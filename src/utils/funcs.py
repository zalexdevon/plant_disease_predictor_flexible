import tensorflow as tf
from Mylib import myfuncs
import os
import pandas as pd
import plotly.express as px


def plot_train_val_scoring_from_trained_models(
    max_val_value, target_val_value, dtick_y_value
):
    plot_dir = "artifacts/plot/components"

    # Get các components
    component_paths = os.listdir(plot_dir)
    components = [
        myfuncs.load_python_object(os.path.join(plot_dir, item))
        for item in component_paths
    ]

    # Vẽ biểu đồ từ các components
    model_names = [item[0] for item in components]
    train_scores = [item[1] for item in components]
    val_scores = [item[2] for item in components]

    for i in range(len(train_scores)):
        if train_scores[i] > max_val_value:
            train_scores[i] = max_val_value

        if val_scores[i] > max_val_value:
            val_scores[i] = max_val_value

    # Vẽ biểu đồ
    df = pd.DataFrame(
        {
            "x": model_names,
            "train": train_scores,
            "val": val_scores,
        }
    )

    df_long = df.melt(
        id_vars=["x"],
        value_vars=["train", "val"],
        var_name="Category",
        value_name="y",
    )

    fig = px.line(
        df_long,
        x="x",
        y="y",
        color="Category",
        markers=True,
        color_discrete_map={
            "train": "gray",
            "val": "blue",
        },
        hover_data={"x": False, "y": True, "Category": False},
    )

    fig.add_hline(
        y=max_val_value,
        line_dash="solid",
        line_color="black",
        line_width=2,
    )

    fig.add_hline(
        y=target_val_value,
        line_dash="dash",
        line_color="green",
        line_width=2,
    )

    fig.update_layout(
        autosize=False,
        width=100 * (len(model_names) + 2) + 30,
        height=400,
        margin=dict(l=30, r=10, t=10, b=0),
        xaxis=dict(
            title="",
            range=[
                0,
                len(model_names) + 2,
            ],
            tickmode="linear",
        ),
        yaxis=dict(
            title="",
            range=[0, max_val_value + dtick_y_value],
            dtick=dtick_y_value,
        ),
        showlegend=False,
    )

    return fig
