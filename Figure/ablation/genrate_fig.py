import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

# Data
strides = np.array([2, 4, 8, 16, 20, 32, 64, 128])
r50 = np.array([0.12838263058527374, 0.12397734424166142, 0.13089993706733793,
                0.1510383889238515, 0.1554436752674638, 0.1592196349905601,
                0.16299559471365638, 0.16299559471365638])
r70 = np.array([0.06293266205160478, 0.05349276274386407, 0.048458149779735685,
                0.05601006922592826, 0.05538074260541221, 0.05601006922592826,
                0.05726872246696035, 0.05726872246696035])

# Set dark background style
plt.style.use('default')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.facecolor': '#2e3037',
    'axes.facecolor': '#2e3037',
    'axes.edgecolor': 'white',
    'grid.color': 'white',
    'grid.alpha': 0.2,
    'grid.linestyle': '--',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.grid': True,
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

# Create figure
fig, ax = plt.subplots(dpi=300)

# Plot with neon-style colors
ax.plot(strides, r50, '-o', color='#00ff8c', linewidth=2.5, 
        label='R@0.5', markerfacecolor='#2e3037', markersize=8, 
        markeredgewidth=2, markeredgecolor='#00ff8c')
ax.plot(strides, r70, '-o', color='#ff5e78', linewidth=2.5, 
        label='R@0.7', markerfacecolor='#2e3037', markersize=8, 
        markeredgewidth=2, markeredgecolor='#ff5e78')

# Set x-axis to log scale and force all stride values to show
ax.set_xscale('log')
ax.set_xticks(strides)  # Set exact tick positions
ax.get_xaxis().set_major_formatter(ScalarFormatter())  # Remove scientific notation
ax.xaxis.set_tick_params(rotation=45)  # Rotate labels for better readability

# Customize spines
for spine in ax.spines.values():
    spine.set_color('white')
    spine.set_linewidth(1.5)

# Set labels
ax.set_xlabel('Stride', labelpad=10)
ax.set_ylabel('Recall', labelpad=10)

# Customize legend
legend = ax.legend(frameon=True, facecolor='#2e3037', edgecolor='white', 
                  loc='upper left', bbox_to_anchor=(0.02, 0.98))
for text in legend.get_texts():
    text.set_color('white')

# Set margins and layout
plt.margins(x=0.05)

# Add a slight glow effect to the lines
from matplotlib.patheffects import withStroke
for line in ax.lines:
    line.set_path_effects([withStroke(linewidth=4, foreground='white', alpha=0.2)])

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save with both formats
plt.savefig('stride_recall_plot_dark.pdf', format='pdf', dpi=300, 
            bbox_inches='tight', pad_inches=0.1, facecolor='#2e3037')
plt.savefig('stride_recall_plot_dark.png', format='png', dpi=300, 
            bbox_inches='tight', pad_inches=0.1, facecolor='#2e3037')

plt.close()
