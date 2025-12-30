import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.interpolate import make_interp_spline
from scipy import stats
import matplotlib as mpl

def create_paper_style_heatmap_with_dashed_line(data_path):
    """Create a correlation heatmap with a custom dashed line next to the color bar"""

    # Verify data path exists
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        print("Please check your file path and try again.")
        return

    # Load your phenotype data
    phenotypes = pd.read_csv(data_path)

    # Check if data has missing values (just for assurance)
    if phenotypes.isnull().any().any():
        print("Warning: Missing data found in the dataset. Proceeding with dropped rows.")
        phenotypes = phenotypes.dropna()

    # Extract the three traits (columns 5,6,7)
    traits = phenotypes.iloc[:, 5:8]
    traits.columns = ['SYPP', 'NPP', 'PYPP']

    # Calculate the correlation matrix
    corr_matrix = traits.corr()

    # Create figure with proper aspect ratio
    plt.figure(figsize=(6, 5))

    # Choose a color palette (you can replace this with any preferred palette)
    cmap = 'coolwarm'  # Change this to any preferred palette

    # Create the heatmap with annotations and custom colors
    ax = sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt=".2f", linewidths=0.5,
                     cbar_kws={'label': ''},  # No label for the color bar
                     annot_kws={'fontsize': 14, 'fontweight': 'bold'})  # Bold font for the numbers inside the heatmap
    ax.text(-0.15, 1.05, 'A', transform=ax.transAxes, fontsize=30, fontweight='bold')

    # Customize font to Times Roman, increase the font size, and make all fonts bold
    plt.xticks(fontsize=12, family='serif', fontweight='bold')  # Bold x-axis label font
    plt.yticks(fontsize=12, family='serif', fontweight='bold')  # Bold y-axis label font

    # Apply bold style for tick labels (fixed error by using labelfontweight instead of labelweight)
    plt.tick_params(axis='both', which='major', labelsize=14)  # Bold tick labels (no 'labelweight' here)

    # Apply bold font style for colorbar ticks
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=12)  # Set font size for color bar ticks
    for tick in colorbar.ax.get_yticklabels():
        tick.set_fontweight('bold')  # Set the font weight to bold for each tick label

    # Add a dashed line next to the color bar with specific dash pattern
    colorbar.ax.plot([1.05, 1.05], [0, 1], linestyle=(0, (5, 1)), color='black', linewidth=2)  # Custom dash pattern

    # Save the heatmap as a PDF with the name "Fig.15A.pdf"
    plt.savefig('Fig.15A.pdf', format='pdf', bbox_inches='tight')
    plt.close()

def create_relationship_visualization(data_path):
    """Create a relationship visualization like Fig. 3 in the MtCro paper with Times New Roman bold font and no title"""

    # Set Times New Roman as the default font
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.weight': 'bold',
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'legend.fontsize': 10,
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'axes.linewidth': 1,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 12  # Increased font size
    })

    # Verify data path exists
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        print("Please check your file path and try again.")
        return

    # Load your phenotype data
    phenotypes = pd.read_csv(data_path)

    # Check if data has missing values
    if phenotypes.isnull().any().any():
        phenotypes = phenotypes.dropna()

    # Extract the three traits (columns 5,6,7)
    traits = phenotypes.iloc[:, 5:8]
    traits.columns = ['SYPP', 'NPP', 'PYPP']

    # Use SYPP and NPP as they have the highest correlation (0.966)
    trait1 = traits['SYPP'].values
    trait2 = traits['NPP'].values

    # 1. Mean-center both traits
    trait1_centered = trait1 - np.mean(trait1)
    trait2_centered = trait2 - np.mean(trait2)

    # 2. Sort by trait1 in descending order
    sort_idx = np.argsort(trait1_centered)[::-1]
    trait1_sorted = trait1_centered[sort_idx]
    trait2_sorted = trait2_centered[sort_idx]

    # 3. Shift one trait for visualization
    # Calculate appropriate shift amount based on data range
    shift_amount = np.mean(np.abs(trait1_sorted)) * 0.8
    trait2_shifted = trait2_sorted - shift_amount

    # 4. Create detailed visualization with MtCro style
    plt.figure(figsize=(10, 6))

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.3, zorder=1)

    # Plot gray dashed lines connecting corresponding points (density visualization)
    for i in range(len(trait1_sorted)):
        plt.plot([i, i], [trait1_sorted[i], trait2_shifted[i]],
                 color='gray', alpha=0.5, linewidth=0.5, zorder=1)

    # Plot every 10th sample as points (as in the paper)
    x_range = range(0, len(trait1_sorted), 10)
    plt.scatter(x_range, trait1_sorted[::10],
                color='green', s=15, zorder=5, label='SYPP')
    plt.scatter(x_range, trait2_shifted[::10],
                color='orange', s=15, zorder=5, label='NPP (shifted)')

    # 5. Add a smoothed curve (not straight line)
    x = np.array(range(len(trait1_sorted)))
    x_new = np.linspace(x.min(), x.max(), 300)

    # Create smooth curves using spline interpolation
    spl1 = make_interp_spline(x, trait1_sorted, k=3)
    spl2 = make_interp_spline(x, trait2_shifted, k=3)
    y1_smooth = spl1(x_new)
    y2_smooth = spl2(x_new)

    # Add the shaded region between the curves
    plt.fill_between(x_new, y1_smooth, y2_smooth, color='gray', alpha=0.15, zorder=2)

    # Calculate regression line for the middle
    midline = (y1_smooth + y2_smooth) / 2
    plt.plot(x_new, midline, 'r-', linewidth=2, zorder=3, label='Fitted Curve')

    # 6. Calculate statistics
    slope, intercept, r_value, p_value, std_err = stats.linregress(trait1_sorted, trait2_shifted)
    r2 = r_value ** 2

    # 7. Format the plot to match MtCro style - NO TITLE
    plt.xlabel('Samples (selected every 10 samples)', fontsize=12)
    plt.ylabel('Value (mean-centered)', fontsize=12)

    # Create custom legend with correct styling
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                   markersize=6, label='SYPP'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                   markersize=6, label='NPP (shifted)'),
        plt.Line2D([0], [0], color='red', linewidth=2, label='Fitted Curve')
    ]
    legend = plt.legend(handles=legend_elements, loc='best')

    # Set tick labels to bold (this must be done after creating the plot)
    plt.xticks(fontweight='bold', fontsize=12)
    plt.yticks(fontweight='bold', fontsize=12)

    # Make legend text bold and Times New Roman
    for text in legend.get_texts():
        text.set_fontweight('bold')
        text.set_fontfamily('Times New Roman')
        text.set_fontsize(11)  # Slightly larger font for legend

    # Set y-axis limits to match the paper's style
    y_min = min(trait1_sorted.min(), trait2_shifted.min()) - 0.5
    y_max = max(trait1_sorted.max(), trait2_shifted.max()) + 0.5
    plt.ylim(y_min, y_max)

    # Adjust margins to match the paper's figure style
    plt.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.1)

    plt.tight_layout()

    # Add the label 'A' in the top left corner, similar to the heatmap style
    ax = plt.gca()  # Get current axes
    ax.text(-0.1, 1.05, 'A', transform=ax.transAxes, fontsize=30, fontweight='bold')

    # Save with high resolution
    plt.savefig('Fig.16A.pdf', format='pdf', bbox_inches='tight')
    plt.close()

    print("Detailed relationship visualization created and saved as 'Fig.16A.pdf'")
    print(f"Correlation: {r_value:.3f}, R-squared: {r2:.3f}")


# Run both functions
if __name__ == "__main__":
    # Users can provide their data path here
    data_path = input("Enter the path to your phenotype data CSV file: ")
    create_paper_style_heatmap_with_dashed_line(data_path)
    create_relationship_visualization(data_path)
