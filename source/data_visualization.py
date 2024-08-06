import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
import bokeh.plotting as bp

def plot_histogram(df, column):
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], bins=10, edgecolor='black', color='skyblue')
    plt.title(f'Histogram of {column}', fontsize=15)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.savefig(f'reports/figures/{column}_histogram.svg')
    plt.close()

def plot_scatter(df, x_col, y_col, hue_col):
    fig = px.scatter(df, x=x_col, y=y_col, color=hue_col,
                     hover_name='first_name', hover_data=['last_name'])
    fig.write_image(f'reports/figures/{x_col}_vs_{y_col}.svg')

def plot_bokeh_scatter(df, x_col, y_col):
    p = bp.figure(title=f"{x_col} vs {y_col}", x_axis_label=x_col, y_axis_label=y_col)
    p.circle(df[x_col], df[y_col], size=10, color="navy", alpha=0.5)
    bp.output_file(f'reports/figures/{x_col}_vs_{y_col}.svg')
    bp.save(p)
