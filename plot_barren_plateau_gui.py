"""
Interactive Matplotlib GUI for Barren Plateau Simulation
Features:
- Exact 1D Cost Landscape slice matching theoretical/empirical benchmarks
- Slow rolling ball (viên bi) smoothly oscillating along the curve
- Real-time Qubit Slider (2 to 14) and Cost Function Radio Buttons (Global vs Local)
- Semi-log Variance plot with full logarithmic sub-grid
Run with: python plot_barren_plateau_gui.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import matplotlib.animation as animation
from scipy.interpolate import pchip_interpolate

plt.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.edgecolor'] = '#94A3B8'
plt.rcParams['axes.linewidth'] = 1.0

# 1. Exact Keypoints from Reference Screenshot
keypoints_x = np.array([
    -1.00, -0.85, -0.55, -0.22, -0.06, 0.00, 0.08, 0.22, 0.38, 0.65, 0.80, 0.88, 1.00
]) * np.pi

keypoints_y = np.array([
    0.64, 0.68, 0.28, 0.59, 0.00, 0.00, 0.00, 0.40, 0.62, 0.36, 0.47, 0.45, 0.63
])

theta_dense = np.linspace(-np.pi, np.pi, 300)
base_landscape = pchip_interpolate(keypoints_x, keypoints_y, theta_dense)
base_landscape = np.clip(base_landscape, 0.0, 1.0)

qubit_list = np.array([2, 4, 6, 8, 10, 12, 14])

def get_variance(n, mode):
    if mode == 'Global':
        return 0.25 * (2.0 ** (-(n - 2)))
    else:
        return 0.25 * ((2.0 / n) ** 1.45)

def get_landscape(n, mode):
    amp = (2.0 ** (-(n - 2) * 0.48)) if mode == 'Global' else ((2.0 / n) ** 0.55)
    c0 = 0.48
    return c0 + (base_landscape - c0) * amp

def get_point_on_landscape(u_theta, n, mode):
    base_val = pchip_interpolate(keypoints_x, keypoints_y, u_theta)
    base_val = np.clip(base_val, 0.0, 1.0)
    amp = (2.0 ** (-(n - 2) * 0.48)) if mode == 'Global' else ((2.0 / n) ** 0.55)
    c0 = 0.48
    return c0 + (base_val - c0) * amp

# 2. Setup Figure & Layout
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 9.4), dpi=120)
fig.patch.set_facecolor('#FFFFFF')
plt.subplots_adjust(left=0.12, right=0.92, top=0.92, bottom=0.22, hspace=0.36)

# --- Top Plot: 1D Cost Landscape Slice ---
line_c, = ax1.plot(theta_dense, get_landscape(2, 'Global'), color='#2563EB', lw=2.8)
fill_c = ax1.fill_between(theta_dense, 0, get_landscape(2, 'Global'), color='#3B82F6', alpha=0.15)

# Rolling Ball (Viên bi)
ball_init_theta = 0.22 * np.pi
ball_init_c = get_point_on_landscape(ball_init_theta, 2, 'Global')
marble, = ax1.plot([ball_init_theta], [ball_init_c], marker='o', markersize=9,
                   markerfacecolor='#2563EB', markeredgecolor='white', markeredgewidth=2.0, zorder=5)

ax1.set_title("1D Cost Landscape Slice C(θ) [Global: O(2⁻ⁿ)]", fontsize=11, fontweight='bold', loc='left', color='#334155', pad=8)
ax1.set_xlim(-np.pi, np.pi)
ax1.set_ylim(-0.02, 1.05)
ax1.set_xticks([-np.pi, 0, np.pi])
ax1.set_xticklabels(['-π', '0', 'π'], fontsize=10.5, fontweight='bold', color='#475569')
ax1.set_yticks([0.0, 0.5, 1.0])
ax1.set_yticklabels(['0.0', '0.5', '1.0'], fontsize=10, color='#475569')
ax1.set_xlabel("Parameter θ", fontsize=10.5, color='#475569')
ax1.axhline(0.5, color='#CBD5E1', linestyle='--', lw=1.0, alpha=0.8)
ax1.grid(True, linestyle=':', alpha=0.4, color='#CBD5E1')

# --- Bottom Plot: Semi-log Variance vs Qubits ---
loc_vars = [get_variance(q, 'Local') for q in qubit_list]
glob_vars = [get_variance(q, 'Global') for q in qubit_list]

ax2.plot(qubit_list, loc_vars, color='#16A34A', linestyle='--', marker='.', lw=2.0, label='Local: O(1/poly(n))')
ax2.plot(qubit_list, glob_vars, color='#2563EB', linestyle='--', marker='.', lw=2.0, label='Global: O(2⁻ⁿ)')
active_marker, = ax2.plot([2], [get_variance(2, 'Global')], marker='o', markersize=12,
                          markerfacecolor='#38BDF8', markeredgecolor='#0F172A', markeredgewidth=2.2, zorder=5)

ax2.set_title("Gradient Variance Var(∂C/∂θ) ↑", fontsize=11, fontweight='bold', loc='left', color='#334155', pad=8)
ax2.set_yscale('log')
ax2.set_xlim(1.5, 14.5)
ax2.set_ylim(5e-6, 1.5)
ax2.set_xticks(qubit_list)
ax2.set_xticklabels([str(q) for q in qubit_list], fontsize=10, color='#475569')
ax2.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
ax2.set_yticklabels(['10µ', '100µ', '1m', '10m', '100m', '1'], fontsize=10, color='#475569')
ax2.set_xlabel("Number of Qubits (n) →", fontsize=10.5, color='#475569', loc='right')
ax2.grid(True, which='both', linestyle='-', alpha=0.35, color='#CBD5E1')
ax2.legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', fontsize=9.5)

# --- KPI Text Displays ---
txt_kpi = fig.text(0.5, 0.165, "VARIANCE Σ²:  0.2500      |      LANDSCAPE SLOPE:  0.500",
                   ha='center', fontsize=11, fontweight='bold', color='#1E293B',
                   bbox=dict(boxstyle='round,pad=0.5', fc='#F8FAFC', ec='#E2E8F0', lw=1.2))

# --- Controls: Slider & Radio ---
ax_slider = plt.axes([0.22, 0.07, 0.35, 0.035], facecolor='#F1F5F9')
slider_n = Slider(ax_slider, 'Number of Qubits', 2, 14, valinit=2, valstep=2,
                  color='#0F172A', valfmt='%d')
slider_n.label.set_fontsize(10)
slider_n.label.set_fontweight('bold')
slider_n.label.set_color('#334155')

ax_radio = plt.axes([0.68, 0.04, 0.22, 0.09], facecolor='#F8FAFC')
radio_cost = RadioButtons(ax_radio, ('Global', 'Local'), active=0, activecolor='#0284C7')
for lbl in radio_cost.labels:
    lbl.set_fontsize(10)
    lbl.set_fontweight('bold')

# State variables
current_n = 2
current_mode = 'Global'

def on_controls_changed(val=None):
    global fill_c, current_n, current_mode
    current_n = int(slider_n.val)
    current_mode = radio_cost.value_selected
    
    y_land = get_landscape(current_n, current_mode)
    line_c.set_ydata(y_land)
    
    fill_c.remove()
    fill_c = ax1.fill_between(theta_dense, 0, y_land, color='#3B82F6', alpha=0.15)
    
    mode_tag = "O(2⁻ⁿ)" if current_mode == 'Global' else "O(1/poly(n))"
    ax1.set_title(f"1D Cost Landscape Slice C(θ) [{current_mode}: {mode_tag}]",
                  fontsize=11, fontweight='bold', loc='left', color='#334155', pad=8)
    
    v = get_variance(current_n, current_mode)
    active_marker.set_data([current_n], [v])
    active_marker.set_markerfacecolor('#38BDF8' if current_mode == 'Global' else '#86EFAC')
    
    slope = np.sqrt(v)
    v_str = f"{v:.4f}" if v >= 1e-3 else f"{v:.2e}"
    s_str = f"{slope:.3f}" if slope >= 1e-3 else f"{slope:.2e}"
    txt_kpi.set_text(f"VARIANCE Σ²:  {v_str}      |      LANDSCAPE SLOPE:  {s_str}")
    
    fig.canvas.draw_idle()

slider_n.on_changed(on_controls_changed)
radio_cost.on_clicked(on_controls_changed)

# --- Slow Smooth Marble Animation (Viên bi dao động chậm rãi) ---
anim_step = 0
def update_marble(frame):
    global anim_step
    anim_step += 1
    # Period = ~120 frames (at 30 fps -> ~4 seconds per full cycle)
    phase = np.sin(anim_step * 2 * np.pi / 120.0)
    
    # Oscillate smoothly between theta = 0.06*pi and 0.38*pi
    cur_theta = (0.22 + 0.16 * phase) * np.pi
    cur_c = get_point_on_landscape(cur_theta, current_n, current_mode)
    
    marble.set_data([cur_theta], [cur_c])
    return marble,

ani = animation.FuncAnimation(fig, update_marble, interval=33, blit=False)

fig.suptitle("Barren Plateau Phenomenon", fontsize=14.5, fontweight='bold', color='#0F172A', y=0.97)

if __name__ == '__main__':
    plt.show()
