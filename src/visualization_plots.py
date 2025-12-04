import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Union
import warnings
import math

# Set style for better looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")


def plot_kde_by_category(df: pd.DataFrame, 
                         value_col: str, 
                         category_col: str,
                         model_col: str = "model_display_name",
                         title: str = None,
                         figsize: tuple = (10, 6),
                         fill: bool = True,
                         common_norm: bool = False,
                         palette: Optional[str] = None,
                         ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot KDE distributions for each category in a single plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique categories
    categories = df[category_col].dropna().unique()
    
    # Plot KDE for each category
    for category in categories:
        subset = df[df[category_col] == category][value_col].dropna()
        if len(subset) > 0:
            sns.kdeplot(data=subset, 
                       label=str(category), 
                       fill=fill,
                       common_norm=common_norm,
                       ax=ax)
    
    # Auto-generate title if not provided
    if title is None:
        model_name = df[model_col].iloc[0] if model_col in df.columns else "Model"
        title = f"{model_name}: {value_col.replace('_', ' ').title()} by {category_col.replace('_', ' ').title()}"
    
    # Customize plot
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(value_col.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    
    # Only add legend if we have categories
    if len(categories) > 0:
        ax.legend(title=category_col.replace('_', ' ').title(), 
                  title_fontsize=11,
                  fontsize=10)
    
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_models_side_by_side(df: pd.DataFrame,
                             value_col: str,
                             category_col: str,
                             model_col: str = "model_display_name",
                             selected_models: List[str] = None,
                             models_per_row: int = 3,
                             figsize_per_plot: tuple = (6, 4),
                             fill: bool = True,
                             sharey: bool = True,
                             sharex: bool = True,
                             palette: str = 'Set2') -> plt.Figure:
    """
    Plot side-by-side KDE plots for multiple models in a dynamic grid layout.
    
    Parameters:
    -----------
    models_per_row : int
        Number of models to display per row (default: 3)
    figsize_per_plot : tuple
        Size of each individual plot (width, height)
    """
    # Get unique models
    if selected_models is None:
        selected_models = df[model_col].dropna().unique()
    else:
        selected_models = [m for m in selected_models if m in df[model_col].values]
    
    n_models = len(selected_models)
    
    # Calculate grid dimensions
    n_rows = math.ceil(n_models / models_per_row)
    n_cols = min(models_per_row, n_models)
    
    # Calculate overall figure size
    fig_width = figsize_per_plot[0] * n_cols
    fig_height = figsize_per_plot[1] * n_rows
    figsize = (fig_width, fig_height)
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=figsize, 
                            sharey=sharey,
                            sharex=sharex,
                            squeeze=False)
    
    # Flatten axes for easy iteration
    axes_flat = axes.flatten()
    
    # Plot each selected model
    for idx, (ax, model_name) in enumerate(zip(axes_flat[:n_models], selected_models)):
        model_df = df[df[model_col] == model_name]
        
        if len(model_df) > 0:
            # Get unique categories for this model
            categories = model_df[category_col].dropna().unique()
            
            # Plot KDE for each category
            for category in categories:
                subset = model_df[model_df[category_col] == category][value_col].dropna()
                if len(subset) > 0:
                    sns.kdeplot(data=subset, 
                               label=str(category), 
                               fill=fill,
                               ax=ax)
            
            # Customize plot
            ax.set_title(model_name, fontsize=12, fontweight='bold')
            ax.set_xlabel(value_col.replace('_', ' ').title() if idx >= n_models - n_cols else "", 
                         fontsize=10)
            ax.set_ylabel('Density' if idx % n_cols == 0 else "", 
                         fontsize=10)
            
            # Add legend only if we have categories
            if len(categories) > 0:
                ax.legend(title='', fontsize=8)
            
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"No data for\n{model_name}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(model_name, fontsize=12, fontweight='bold')
            ax.set_xlabel(value_col.replace('_', ' ').title(), fontsize=10)
    
    # Hide unused axes
    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    # Add overall title
    plt.suptitle(f"{value_col.replace('_', ' ').title()} Distribution by {category_col.replace('_', ' ').title()}",
                fontsize=16,
                fontweight='bold',
                y=1.02)
    
    plt.tight_layout()
    return fig


def plot_comparison_grid(df: pd.DataFrame,
                         value_col: str,
                         category_cols: List[str],
                         model_col: str = "model_display_name",
                         selected_models: List[str] = None,
                         models_per_row: int = 3,
                         figsize_per_plot: tuple = (6, 4)) -> plt.Figure:
    """
    Create a grid of plots comparing distributions across models and multiple grouping variables.
    """
    # Get unique models
    if selected_models is None:
        selected_models = df[model_col].dropna().unique()
    else:
        selected_models = [m for m in selected_models if m in df[model_col].values]
    
    n_models = len(selected_models)
    n_categories = len(category_cols)
    
    # Calculate grid dimensions for models
    n_model_rows = math.ceil(n_models / models_per_row)
    n_model_cols = min(models_per_row, n_models)
    
    # Overall grid: categories × models
    fig_width = figsize_per_plot[0] * n_model_cols
    fig_height = figsize_per_plot[1] * n_model_rows * n_categories
    figsize = (fig_width, fig_height)
    
    # Create subplot grid: categories (rows) × models (cols)
    fig, axes = plt.subplots(n_categories * n_model_rows, n_model_cols,
                            figsize=figsize,
                            squeeze=False)
    
    # Plot grid
    for cat_idx, category_col in enumerate(category_cols):
        for model_idx, model_name in enumerate(selected_models):
            # Calculate position in grid
            row_offset = cat_idx * n_model_rows + model_idx // n_model_cols
            col_offset = model_idx % n_model_cols
            
            ax = axes[row_offset, col_offset]
            model_df = df[df[model_col] == model_name]
            
            if len(model_df) > 0:
                # Get unique categories for this model and column
                categories = model_df[category_col].dropna().unique()
                
                # Plot KDE for each category
                for category in categories:
                    subset = model_df[model_df[category_col] == category][value_col].dropna()
                    if len(subset) > 0:
                        sns.kdeplot(data=subset, 
                                   label=str(category), 
                                   fill=True,
                                   ax=ax)
                
                # Set titles and labels
                if row_offset == 0:
                    ax.set_title(model_name, fontsize=10, fontweight='bold')
                
                if col_offset == 0:
                    category_label = category_col.replace('_', ' ').title()
                    if model_idx == 0:
                        ax.set_ylabel(f"{category_label}", fontsize=10)
                    else:
                        ax.set_ylabel(f"{category_label}\n(Model {model_idx // n_model_cols + 1})", fontsize=9)
                
                # Only show x-label on bottom row
                if row_offset == (n_categories * n_model_rows - 1) or \
                   (cat_idx == n_categories - 1 and model_idx // n_model_cols == n_model_rows - 1):
                    ax.set_xlabel(value_col.replace('_', ' ').title(), fontsize=10)
                
                ax.legend(title='', fontsize=6)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f"No data", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=8)
    
    # Hide unused axes
    total_cells = n_categories * n_model_rows * n_model_cols
    used_cells = n_categories * n_models
    
    if used_cells < total_cells:
        for idx in range(used_cells, total_cells):
            row = idx // n_model_cols
            col = idx % n_model_cols
            axes[row, col].axis('off')
    
    plt.suptitle(f"Distribution of {value_col.replace('_', ' ').title()} by Demographic Factors",
                fontsize=14,
                fontweight='bold',
                y=1.02)
    plt.tight_layout()
    return fig


def create_demographic_comparisons(df: pd.DataFrame,
                                   value_col: str = "parsed_urgency_score",
                                   model_col: str = "model_display_name",
                                   demographic_cols: List[str] = None,
                                   models_per_row: int = 3,
                                   figsize_per_plot: tuple = (6, 4)) -> Dict[str, plt.Figure]:
    """
    Create comparison figures for each demographic factor with dynamic grid layout.
    
    Returns:
    --------
    dict: Dictionary with figure names as keys and matplotlib figures as values
    """
    if demographic_cols is None:
        demographic_cols = ['gender', 'language', 'occupation', 'race_ethnicity']
    
    # Filter to only include columns that exist
    demographic_cols = [col for col in demographic_cols if col in df.columns]
    
    # Get unique models
    all_models = df[model_col].dropna().unique()
    
    print(f"Found {len(all_models)} models")
    print(f"Creating comparisons for: {', '.join(demographic_cols)}")
    
    figures = {}
    
    for demo_col in demographic_cols:
        print(f"\nCreating plots for '{demo_col}':")
        
        # Check data availability
        non_null_count = df[demo_col].notna().sum()
        unique_values = df[demo_col].nunique()
        
        if non_null_count == 0:
            print(f"  Skipping: No data")
            continue
        
        # Count models that have data for this demographic
        models_with_data = []
        for model in all_models:
            model_demo_data = df[(df[model_col] == model) & (df[demo_col].notna())]
            if len(model_demo_data) > 0:
                models_with_data.append(model)
        
        if len(models_with_data) == 0:
            print(f"  Skipping: No model has data for this demographic")
            continue
        
        print(f"  Models with data: {len(models_with_data)}/{len(all_models)}")
        print(f"  Unique values: {unique_values}")
        
        # Create figure
        fig = plot_models_side_by_side(
            df=df,
            value_col=value_col,
            category_col=demo_col,
            model_col=model_col,
            selected_models=models_with_data,
            models_per_row=models_per_row,
            figsize_per_plot=figsize_per_plot
        )
        
        figures[demo_col] = fig
    
    return figures


def plot_single_model_comparison(df: pd.DataFrame,
                                 value_col: str,
                                 model_col: str = "model_display_name",
                                 selected_models: List[str] = None,
                                 models_per_row: int = 3,
                                 figsize_per_plot: tuple = (6, 4)) -> plt.Figure:
    """
    Plot comparison of value distribution across multiple models (no demographic breakdown).
    """
    if selected_models is None:
        selected_models = df[model_col].dropna().unique()
    
    n_models = len(selected_models)
    n_rows = math.ceil(n_models / models_per_row)
    n_cols = min(models_per_row, n_models)
    
    fig_width = figsize_per_plot[0] * n_cols
    fig_height = figsize_per_plot[1] * n_rows
    figsize = (fig_width, fig_height)
    
    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=figsize,
                            sharey=True,
                            squeeze=False)
    
    axes_flat = axes.flatten()
    
    for idx, (ax, model_name) in enumerate(zip(axes_flat[:n_models], selected_models)):
        model_data = df[df[model_col] == model_name][value_col].dropna()
        
        if len(model_data) > 0:
            sns.kdeplot(data=model_data, fill=True, ax=ax)
            ax.set_title(model_name, fontsize=12, fontweight='bold')
            ax.set_xlabel(value_col.replace('_', ' ').title(), fontsize=10)
            ax.set_ylabel('Density' if idx % n_cols == 0 else "", fontsize=10)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"No data", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model_name, fontsize=12, fontweight='bold')
    
    # Hide unused axes
    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.suptitle(f"{value_col.replace('_', ' ').title()} Distribution by Model",
                fontsize=16,
                fontweight='bold',
                y=1.02)
    plt.tight_layout()
    return fig


# Example usage
if __name__ == "__main__":
    print("Visualization Plotting Module (Dynamic Grid Layout)")
    print("=" * 60)
    print("Key features:")
    print("• Dynamic grid layout based on number of models")
    print("• Configurable models per row (default: 3)")
    print("• Automatic figure sizing")
    print("• Clean handling of empty plots")