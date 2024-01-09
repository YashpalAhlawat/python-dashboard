# pylint: disable=E1101, W0621, W0102
"""
Plotly Dashboard for Wine Analysis
"""

import pandas as pd
import plotly.express as px
from sklearn.datasets import load_wine
from dash import Dash, html, dcc
from dash.dependencies import Input, Output


# Load Data
def load_wine_data_as_dataframe():
    """
    Load the wine dataset and create a pandas DataFrame.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame containing the wine dataset with features and WineType column.
    """
    wine_dataset = load_wine()
    features_df = pd.DataFrame(wine_dataset.data, columns=wine_dataset.feature_names)
    wine_types = [wine_dataset.target_names[t] for t in wine_dataset.target]
    features_df["WineType"] = wine_types
    return features_df


wine_df = load_wine_data_as_dataframe()
ingredients = wine_df.drop(columns=["WineType"]).columns

avg_wine_df = wine_df.groupby("WineType").mean().reset_index()


def create_scatter_chart(x_axis="alcohol", y_axis="malic_acid", color_encode=False):
    """
    Create a scatter chart using Plotly Express.

    Parameters:
    -----------
    x_axis: str
        Feature for the x-axis.
    y_axis: str
        Feature for the y-axis.
    color_encode: bool
        Whether to color-encode the chart.

    Returns:
    --------
    plotly.graph_objects.Figure
        Scatter chart figure.
    """
    scatter_fig = px.scatter(wine_df, x=x_axis, y=y_axis,
                             color="WineType" if color_encode else None,
                             title=f"{x_axis.capitalize()} vs {y_axis.capitalize()}")
    return scatter_fig


def create_bar_chart(ingredients=["alcohol", "malic_acid", "ash"]):
    """
    Create a bar chart using Plotly Express.

    Parameters:
    -----------
    ingredients: list
        List of ingredients to plot on the y-axis.

    Returns:
    --------
    plotly.graph_objects.Figure
        Bar chart figure.
    """
    bar_fig = px.bar(avg_wine_df, x="WineType", y=ingredients,
                     title="Avg. Ingredients per Wine Type")
    bar_fig.update_layout(height=600)
    return bar_fig


# Create widgets
x_axis = dcc.Dropdown(id="x_axis", options=[{'label': col, 'value': col} for col in ingredients],
                      value="alcohol", clearable=False,
                      style={"display": "inline-block", "width": "49%"})
y_axis = dcc.Dropdown(id="y_axis", options=[{'label': col, 'value': col} for col in ingredients],
                      value="malic_acid", clearable=False,
                      style={"display": "inline-block", "width": "49%"})
color_encode = dcc.Checklist(id="color_encode", options=[{'label': "Color-Encode", 'value': True}],
                             value=[], inline=True)

multi_select = dcc.Dropdown(id="multi_select",
                            options=[{'label': col, 'value': col} for col in ingredients],
                            value=["alcohol", "malic_acid", "ash"], clearable=False, multi=True)

# Web app layout
app = Dash(title="Wine Analysis")

app.layout = html.Div(
    children=[
        html.H1("Wine Analysis Dashboard", style={"text-align": "center"}),
        html.Div("Explore the relationship between various ingredients used in "
                 "the creation of three different types of Wines (class_0, class_1, class_2)",
                 style={"text-align": "center"}),
        html.Br(),
        html.Div(
            children=[
                x_axis, y_axis, color_encode,
                dcc.Graph(id="scatter_chart", figure=create_scatter_chart())
            ],
            style={"display": "inline-block", "width": "49%"}
        ),
        html.Div(
            children=[
                multi_select, html.Br(),
                dcc.Graph(id="bar_chart", figure=create_bar_chart())
            ],
            style={"display": "inline-block", "width": "49%"}
        )
    ],
    style={"padding": "50px"}
)


# Callbacks to join widgets with filters
@app.callback(Output("scatter_chart", "figure"),
              [Input("x_axis", "value"),
               Input("y_axis", "value"),
               Input("color_encode", "value")])
def update_scatter_chart(x_axis, y_axis, color_encode):
    """
    Callback function to update the scatter chart based on user inputs.

    Parameters:
    -----------
    x_axis: str
        Feature for the x-axis.
    y_axis: str
        Feature for the y-axis.
    color_encode: list
        List of color-encoding options.

    Returns:
    --------
    plotly.graph_objects.Figure
        Updated scatter chart figure.
    """
    return create_scatter_chart(x_axis, y_axis, color_encode)


@app.callback(Output("bar_chart", "figure"),
              [Input("multi_select", "value")])
def update_bar_chart(ingredients):
    """
    Callback function to update the bar chart based on user-selected ingredients.

    Parameters:
    -----------
    ingredients: list
        List of ingredients to be plotted on the y-axis of the bar chart.

    Returns:
    --------
    plotly.graph_objects.Figure
        Updated bar chart figure.
    """
    return create_bar_chart(ingredients)


if __name__ == '__main__':
    app.run_server(debug=True)
