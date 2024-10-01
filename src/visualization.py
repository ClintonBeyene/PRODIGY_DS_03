import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to plot pie chart
def plot_pie_chart(data, column, ax, colors, labels, title, ylabel):
    """
    Plots a pie chart.
    
    Parameters:
    - data: DataFrame containing the data
    - column: Column to plot
    - ax: Axis to plot on
    - colors: List of colors for the pie chart
    - labels: List of labels for the pie chart
    - title: Title of the pie chart
    - ylabel: Y-axis label
    """
    data[column].value_counts().plot.pie(
        explode=[0, 0.25], autopct='%1.2f%%', ax=ax, shadow=True, colors=colors, 
        labels=labels, fontsize=12, startangle=25
    )
    ax.set_title(title, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=14)

# Function to plot bar chart
def plot_bar_chart(data, x_col, y_col, hue_col, ax, palette, ylabel):
    """
    Plots a bar chart.
    
    Parameters:
    - data: DataFrame containing the data
    - x_col: Column for x-axis
    - y_col: Column for y-axis
    - hue_col: Column for hue
    - ax: Axis to plot on
    - palette: List of colors for the bars
    - ylabel: Y-axis label
    """
    sns.barplot(
        x=x_col, y=y_col, hue=hue_col, data=data, palette=palette, 
        estimator=lambda x: len(x) / len(data) * 100, ax=ax
    )
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticklabels(data[x_col].unique(), rotation=0, rotation_mode="anchor")
    ax.legend(title=hue_col, loc='upper right')


# Function to set the style
def set_plot_style(style='whitegrid'):
    sns.set_style(style)

# Function to create subplots
def create_subplots(nrows, ncols, figsize=(15, 15)):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    return fig, axes.flatten()

# Function to plot histograms
def plot_histograms(df, numerical_columns, axes, fig, bins=30):
    for i, column in enumerate(numerical_columns):
        sns.histplot(df[column], kde=True, ax=axes[i], bins=bins)
        axes[i].set_title(f'Distribution of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True)
    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

# Function to adjust layout and show plot
def show_plot():
    plt.tight_layout()
    plt.show()

# Main function to call the above functions
def numerical_distribution(df, numerical_columns):
    set_plot_style()
    fig, axes = create_subplots(nrows=3, ncols=3)
    plot_histograms(df, numerical_columns, axes, fig)
    show_plot()