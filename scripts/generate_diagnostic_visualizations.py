"""
Tạo 4 bộ trực quan hóa ĐỘC LẬP (riêng cho từng mục, không gộp chung):
1. Hội tụ (Convergence)
2. Cực tiểu cục bộ (Local Minimum)
3. Tầng chết (Dead Layer)
4. Bình nguyên cằn cỗi (Barren Plateau)

Mỗi mục gồm:
- File Matplotlib (.png) độ phân giải cao (300 DPI), typography đẹp cho slide/báo cáo.
- File Plotly Interactive Animated (.html) có thể tương tác 3D, thanh trượt thời gian, nút Play/Pause và Bảng HUD Viễn trắc.
"""
import os
import sys
import numpy as np

if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Tạo thư mục đầu ra
OUT_DIR = "figures/diagnostics"
os.makedirs(OUT_DIR, exist_ok=True)

# Cấu hình Matplotlib thẩm mỹ
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#4B5563'
plt.rcParams['axes.linewidth'] = 1.0

# ==============================================================================
# 1. HỘI TỤ (TRUE CONVERGENCE)
# ==============================================================================
def build_convergence():
    print("-> Đang tạo Trực quan hóa 1: HỘI TỤ (True Convergence)...")
    
    # 1.1 Matplotlib PNG
    theta1 = np.linspace(-2.5, 2.5, 200)
    theta2 = np.linspace(-2.5, 2.5, 200)
    T1, T2 = np.meshgrid(theta1, theta2)
    # Hàm mất mát dạng Elip mượt mà dẫn tới đáy (0, 0)
    Loss = 0.04 + 0.18 * (T1**2) + 0.12 * (T2**2) + 0.05 * np.sin(1.5 * T1) * np.sin(1.5 * T2)
    
    # Quỹ đạo tối ưu từ (2.0, 1.8) về (0.0, 0.0)
    steps = 30
    t = np.linspace(0, 1, steps)
    traj_x = 2.0 * np.exp(-3.5 * t) * np.cos(1.2 * t)
    traj_y = 1.8 * np.exp(-3.0 * t) * np.cos(0.8 * t)
    traj_loss = 0.04 + 0.18 * (traj_x**2) + 0.12 * (traj_y**2)
    
    fig = plt.figure(figsize=(14, 6), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1.0])
    
    # Subplot 1: 3D Surface
    ax1 = fig.add_subplot(gs[0], projection='3d')
    surf = ax1.plot_surface(T1, T2, Loss, cmap='viridis_r', alpha=0.85, edgecolor='none')
    ax1.plot(traj_x, traj_y, traj_loss, color='#EF4444', linewidth=3, label='Quỹ đạo tối ưu (Adam)')
    ax1.scatter([traj_x[0]], [traj_y[0]], [traj_loss[0]], color='#F59E0B', s=80, label='Khởi đầu (Loss=0.85)')
    ax1.scatter([traj_x[-1]], [traj_y[-1]], [traj_loss[-1]], color='#10B981', s=120, marker='*', label='Đáy toàn cục (Loss=0.04)')
    ax1.set_title("Cảnh quan Hàm mất mát 3D (Thung lũng toàn cục)", fontsize=13, fontweight='bold', pad=15)
    ax1.set_xlabel("Tham số $\\theta_1$", labelpad=8)
    ax1.set_ylabel("Tham số $\\theta_2$", labelpad=8)
    ax1.set_zlabel("Validation Loss $\\mathcal{L}$", labelpad=8)
    ax1.view_init(elev=32, azim=225)
    
    # Subplot 2: 2D Contour + Gradient Vectors + Telemetry
    ax2 = fig.add_subplot(gs[1])
    cs = ax2.contourf(T1, T2, Loss, levels=25, cmap='viridis_r', alpha=0.8)
    plt.colorbar(cs, ax=ax2, label='Validation Loss')
    ax2.plot(traj_x, traj_y, color='#EF4444', linewidth=2.5, linestyle='--', label='Đường trượt dốc')
    
    # Vẽ vector gradient co lại khi tới gần đáy
    idx_arrows = [0, 4, 9, 15, 22]
    for idx in idx_arrows:
        gx = - (0.36 * traj_x[idx])
        gy = - (0.24 * traj_y[idx])
        grad_norm = np.sqrt(gx**2 + gy**2)
        ax2.arrow(traj_x[idx], traj_y[idx], gx*0.7, gy*0.7, head_width=0.12*grad_norm, head_length=0.15*grad_norm,
                  fc='#FBBF24', ec='#B45309', linewidth=1.5, zorder=5)
    
    ax2.scatter(traj_x[-1], traj_y[-1], color='#10B981', s=180, marker='*', edgecolors='black', zorder=6)
    ax2.text(traj_x[-1]+0.2, traj_y[-1]-0.25, "Đáy toàn cục\n$\\nabla \\mathcal{L} \\to 0$, Loss=0.04", 
             fontsize=10, fontweight='bold', color='#065F46', bbox=dict(boxstyle="round,pad=0.3", fc="#D1FAE5", ec="#10B981"))
    
    # Hộp viễn trắc Telemetry HUD
    hud_text = (
        "CHẨN ĐOÁN VIỄN TRẮC (TELEMETRY):\n"
        "• Loss mục tiêu: 0.04 (ĐẠT - Solved!)\n"
        "• Loss Slope: 0.0000 (Dừng dốc)\n"
        "• Gradient Norm: 1.25 -> 0.0001\n"
        "• Bản chất: Cực tiểu toàn cục, dốc hết dốc\n"
        "• Hành động: LOCK_GROWTH (Bảo toàn mạch)"
    )
    ax2.text(0.03, 0.96, hud_text, transform=ax2.transAxes, fontsize=9.5, verticalalignment='top',
             fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", fc="#1F2937", ec="#374151", alpha=0.9), color="#F9FAFB")
    
    ax2.set_title("Mặt phẳng đẳng mức (Contour) & Vector Gradient", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Tham số $\\theta_1$")
    ax2.set_ylabel("Tham số $\\theta_2$")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/01_convergence.png", dpi=300)
    plt.close()
    
    # 1.2 Plotly Interactive Animated HTML
    frames = []
    for k in range(steps):
        cur_x = traj_x[:k+1]
        cur_y = traj_y[:k+1]
        cur_z = traj_loss[:k+1]
        cur_norm = np.sqrt((0.36*traj_x[k])**2 + (0.24*traj_y[k])**2)
        
        frame = go.Frame(
            data=[
                go.Surface(x=theta1, y=theta2, z=Loss, colorscale='Viridis', reversescale=True, opacity=0.85, showscale=False),
                go.Scatter3d(x=cur_x, y=cur_y, z=cur_z, mode='lines+markers', line=dict(color='#EF4444', width=6),
                             marker=dict(size=4, color='#EF4444')),
                go.Scatter3d(x=[traj_x[k]], y=[traj_y[k]], z=[traj_loss[k]], mode='markers+text',
                             marker=dict(size=10, color='#F59E0B', symbol='diamond'),
                             text=[f"Epoch {k+1}: Loss={traj_loss[k]:.3f} | |∇L|={cur_norm:.3f}"],
                             textposition="top center")
            ],
            name=f"frame_{k}"
        )
        frames.append(frame)
        
    fig_plotly = go.Figure(
        data=[
            go.Surface(x=theta1, y=theta2, z=Loss, colorscale='Viridis', reversescale=True, opacity=0.85, colorbar=dict(title="Loss")),
            go.Scatter3d(x=[traj_x[0]], y=[traj_y[0]], z=[traj_loss[0]], mode='lines+markers', line=dict(color='#EF4444', width=6),
                         name='Quỹ đạo tối ưu'),
            go.Scatter3d(x=[traj_x[0]], y=[traj_y[0]], z=[traj_loss[0]], mode='markers+text',
                         marker=dict(size=10, color='#F59E0B'), name='Viên bi tham số',
                         text=[f"Epoch 1: Loss={traj_loss[0]:.3f}"], textposition="top center")
        ],
        layout=go.Layout(
            title="<b>1. HIỆN TƯỢNG HỘI TỤ (TRUE CONVERGENCE)</b><br><sup>Gradient tiến về 0 vì đã chạm đáy thung lũng toàn cục (Loss cực thấp)</sup>",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=12),
            scene=dict(
                xaxis_title="Tham số θ₁", yaxis_title="Tham số θ₂", zaxis_title="Validation Loss",
                camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.2))
            ),
            updatemenus=[dict(
                type="buttons", showactive=False, x=0.05, y=0.05,
                buttons=[
                    dict(label="▶ Play Animation", method="animate", args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True)]),
                    dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                ]
            )],
            sliders=[dict(
                steps=[dict(method="animate", args=[[f"frame_{k}"], dict(mode="immediate", frame=dict(duration=0, redraw=True))], label=f"Ep {k+1}") for k in range(steps)],
                x=0.2, y=0.05, len=0.75
            )]
        ),
        frames=frames
    )
    fig_plotly.write_html(f"{OUT_DIR}/01_convergence.html")
    print("   [OK] Đã lưu: figures/diagnostics/01_convergence.png & .html")

# ==============================================================================
# 2. CỰC TIỂU CỤC BỘ (LOCAL MINIMUM)
# ==============================================================================
def build_local_minimum():
    print("-> Đang tạo Trực quan hóa 2: CỰC TIỂU CỤC BỘ (Local Minimum)...")
    
    # 2.1 Matplotlib PNG
    theta1 = np.linspace(-2.8, 2.8, 250)
    theta2 = np.linspace(-2.0, 2.0, 200)
    T1, T2 = np.meshgrid(theta1, theta2)
    
    # Hàm Double-well chuẩn mực:
    # Hố nông cục bộ tại (-1.6, 0.0) có Loss = 0.528 (Lưng chừng núi)
    # Đáy sâu toàn cục tại (+1.6, 0.0) có Loss = 0.048 (Đáy vực sâu)
    # Vách ngăn thế năng ở giữa x=0 cao ~0.80
    base = 0.90 + 0.05 * (T1**2 + T2**2)
    w_glob = - 0.98 * np.exp(-((T1 - 1.6)**2 + 1.2*T2**2) / (2 * 0.70**2))
    w_loc  = - 0.50 * np.exp(-((T1 + 1.6)**2 + 1.2*T2**2) / (2 * 0.70**2))
    Loss = base + w_glob + w_loc
    
    # Quỹ đạo lăn từ trên cao (-2.5, 1.0) và bị KẸT VÀO HỐ NÔNG tại (-1.6, 0.0)
    steps = 30
    t = np.linspace(0, 1, steps)
    traj_x = -2.5 + 0.9 * (1 - np.exp(-3.5 * t))
    traj_y = 1.0 * np.exp(-3.5 * t)
    traj_base = 0.90 + 0.05 * (traj_x**2 + traj_y**2)
    traj_glob = - 0.98 * np.exp(-((traj_x - 1.6)**2 + 1.2*traj_y**2) / (2 * 0.70**2))
    traj_loc  = - 0.50 * np.exp(-((traj_x + 1.6)**2 + 1.2*traj_y**2) / (2 * 0.70**2))
    traj_loss = traj_base + traj_glob + traj_loc
    
    fig = plt.figure(figsize=(14, 6), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1.0])
    
    ax1 = fig.add_subplot(gs[0], projection='3d')
    surf = ax1.plot_surface(T1, T2, Loss, cmap='Spectral_r', alpha=0.88, edgecolor='none')
    ax1.plot(traj_x, traj_y, traj_loss, color='#DC2626', linewidth=3.5, label='Kẹt trong Hố Cục Bộ')
    ax1.scatter([traj_x[-1]], [traj_y[-1]], [traj_loss[-1]], color='#DC2626', s=130, marker='X', 
                label=f'Kẹt (Loss={traj_loss[-1]:.2f}, ∇L=0)', depthshade=False)
    ax1.scatter([1.6], [0.0], [0.048], color='#10B981', s=160, marker='*', 
                label='Đáy toàn cục (Loss=0.05)', depthshade=False)
    
    ax1.set_title("Cảnh quan Hàm mất mát 3D (Bẫy Cực Tiểu Cục Bộ)", fontsize=13, fontweight='bold', pad=15)
    ax1.set_xlabel("Tham số $\\theta_1$", labelpad=8)
    ax1.set_ylabel("Tham số $\\theta_2$", labelpad=8)
    ax1.set_zlabel("Validation Loss $\\mathcal{L}$", labelpad=8)
    ax1.set_zlim(0.0, 1.4)
    ax1.view_init(elev=32, azim=225)
    ax1.legend(loc='upper right', fontsize=8.5)
    
    ax2 = fig.add_subplot(gs[1])
    cs = ax2.contourf(T1, T2, Loss, levels=25, cmap='Spectral_r', alpha=0.85)
    plt.colorbar(cs, ax=ax2, label='Validation Loss')
    ax2.plot(traj_x, traj_y, color='#DC2626', linewidth=2.5, linestyle='--')
    ax2.scatter(traj_x[-1], traj_y[-1], color='#DC2626', s=200, marker='X', edgecolors='black', zorder=6)
    ax2.scatter(1.6, 0.0, color='#10B981', s=220, marker='*', edgecolors='black', zorder=6)
    
    # Mũi tên giải cứu khi Add Layer (Chỉ rõ đường thoát từ hố nông sang đáy sâu)
    ax2.annotate('Lối thoát khi ADD_LAYER\n(Tăng chiều vượt rào cản thế năng)', xy=(1.6, 0.0), xytext=(-0.3, -1.5),
                 arrowprops=dict(facecolor='#2563EB', shrink=0.08, width=2.2, headwidth=8.5),
                 fontsize=9.2, fontweight='bold', color='#1E40AF',
                 bbox=dict(boxstyle="round,pad=0.3", fc="#DBEAFE", ec="#2563EB"))
    
    ax2.text(traj_x[-1]-0.1, traj_y[-1]+0.35, "HỐ NÔNG CỤC BỘ!\nLoss = 0.52 (Cao)\n$\\nabla \\mathcal{L} = 0$",
             fontsize=9.2, fontweight='bold', color='#991B1B', ha='center',
             bbox=dict(boxstyle="round,pad=0.3", fc="#FEE2E2", ec="#DC2626"))
    
    ax2.text(1.6, 0.35, "ĐÁY TOÀN CỤC\nLoss = 0.05 (Thấp)\nĐích tối ưu",
             fontsize=9.2, fontweight='bold', color='#065F46', ha='center',
             bbox=dict(boxstyle="round,pad=0.3", fc="#D1FAE5", ec="#10B981"))
    
    hud_text = (
        "CHẨN ĐOÁN VIỄN TRẮC (TELEMETRY):\n"
        "• Loss hiện tại: 0.52 (CHƯA ĐẠT - Hố nông!)\n"
        "• Loss Slope: 0.0001 (Đi ngang - Stall)\n"
        "• Gradient Norm: Triệt tiêu CỤC BỘ tại hố\n"
        "• Bản chất: Bị chặn bởi rào cản thế năng ~0.80\n"
        "• Hành động: ADD_LAYER (Cấy tầng mở chiều)"
    )
    ax2.text(0.03, 0.96, hud_text, transform=ax2.transAxes, fontsize=8.8, verticalalignment='top',
             fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.4", fc="#1F2937", ec="#374151", alpha=0.9), color="#F9FAFB")
    
    ax2.set_title("Mặt cắt Đẳng mức: Hố Nông vs Đáy Sâu Toàn Cục", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Tham số $\\theta_1$")
    ax2.set_ylabel("Tham số $\\theta_2$")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/02_local_minimum.png", dpi=300)
    plt.close()
    
    # 2.2 Plotly Interactive Animated HTML
    frames = []
    for k in range(steps):
        cur_x = traj_x[:k+1]
        cur_y = traj_y[:k+1]
        cur_z = traj_loss[:k+1]
        
        frame = go.Frame(
            data=[
                go.Surface(x=theta1, y=theta2, z=Loss, colorscale='Spectral', reversescale=True, opacity=0.85, showscale=False),
                go.Scatter3d(x=cur_x, y=cur_y, z=cur_z, mode='lines+markers', line=dict(color='#DC2626', width=6),
                             marker=dict(size=4, color='#DC2626')),
                go.Scatter3d(x=[traj_x[k]], y=[traj_y[k]], z=[traj_loss[k]], mode='markers+text',
                             marker=dict(size=11, color='#DC2626', symbol='x'),
                             text=[f"Epoch {k+1}: Kẹt ở Hố Nông Loss={traj_loss[k]:.3f} | ∇L≈0"],
                             textposition="top center"),
                go.Scatter3d(x=[1.6], y=[0.0], z=[0.048], mode='markers+text',
                             marker=dict(size=12, color='#10B981', symbol='diamond'),
                             text=["Đáy sâu Toàn cục (Loss=0.05)"], textposition="bottom center")
            ],
            name=f"frame_{k}"
        )
        frames.append(frame)
        
    fig_plotly = go.Figure(
        data=[
            go.Surface(x=theta1, y=theta2, z=Loss, colorscale='Spectral', reversescale=True, opacity=0.85, colorbar=dict(title="Loss")),
            go.Scatter3d(x=[traj_x[0]], y=[traj_y[0]], z=[traj_loss[0]], mode='lines+markers', line=dict(color='#DC2626', width=6),
                         name='Quỹ đạo bị bẫy'),
            go.Scatter3d(x=[traj_x[0]], y=[traj_y[0]], z=[traj_loss[0]], mode='markers+text',
                         marker=dict(size=11, color='#DC2626'), name='Viên bi',
                         text=[f"Epoch 1: Loss={traj_loss[0]:.3f}"], textposition="top center"),
            go.Scatter3d(x=[1.6], y=[0.0], z=[0.048], mode='markers+text',
                         marker=dict(size=12, color='#10B981'), name='Đích toàn cục',
                         text=["Đáy sâu Toàn cục (Loss=0.05)"], textposition="bottom center")
        ],
        layout=go.Layout(
            title="<b>2. HIỆN TƯỢNG CỰC TIỂU CỤC BỘ (LOCAL MINIMUM)</b><br><sup>Gradient tiến về 0 vì kẹt trong hố nông (Loss=0.52 cao hơn hẳn đáy sâu 0.05)</sup>",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=12),
            scene=dict(
                xaxis_title="Tham số θ₁", yaxis_title="Tham số θ₂", zaxis_title="Validation Loss",
                zaxis=dict(range=[0.0, 1.4]),
                camera=dict(eye=dict(x=-1.6, y=-1.6, z=1.3))
            ),
            updatemenus=[dict(
                type="buttons", showactive=False, x=0.05, y=0.05,
                buttons=[
                    dict(label="▶ Play Animation", method="animate", args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True)]),
                    dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                ]
            )],
            sliders=[dict(
                steps=[dict(method="animate", args=[[f"frame_{k}"], dict(mode="immediate", frame=dict(duration=0, redraw=True))], label=f"Ep {k+1}") for k in range(steps)],
                x=0.2, y=0.05, len=0.75
            )]
        ),
        frames=frames
    )
    fig_plotly.write_html(f"{OUT_DIR}/02_local_minimum.html")
    print("   [OK] Đã lưu: figures/diagnostics/02_local_minimum.png & .html")

# ==============================================================================
# 3. TẦNG CHẾT (DEAD LAYER)
# ==============================================================================
def build_dead_layer():
    print("-> Đang tạo Trực quan hóa 3: TẦNG CHẾT (Dead Layer)...")
    
    # 3.1 Matplotlib PNG
    fig = plt.figure(figsize=(15, 6), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1.1])
    
    # Subplot 1: Cảnh quan tham số cắt lớp 3D (Tầng 1 dốc vs Tầng 2 phẳng lì)
    theta1 = np.linspace(-np.pi, np.pi, 200) # Tầng 1 (Active)
    theta2 = np.linspace(-np.pi, np.pi, 200) # Tầng 2 (Dead)
    T1, T2 = np.meshgrid(theta1, theta2)
    # Loss biến thiên mạnh theo theta1, nhưng HOÀN TOÀN BẤT BIẾN theo theta2
    Loss = 0.25 + 0.35 * np.cos(T1) + 0.002 * np.sin(T2)
    
    ax1 = fig.add_subplot(gs[0], projection='3d')
    surf = ax1.plot_surface(T1, T2, Loss, cmap='coolwarm', alpha=0.85, edgecolor='none')
    ax1.set_title("Cảnh quan Loss: Tầng 1 (Active) vs Tầng 2 (Dead)", fontsize=12, fontweight='bold', pad=15)
    ax1.set_xlabel("Góc quay Tầng 1 ($\\theta^{(1)}$)", labelpad=8)
    ax1.set_ylabel("Góc quay Tầng 2 ($\\theta^{(2)}$)", labelpad=8)
    ax1.set_zlabel("Validation Loss", labelpad=8)
    ax1.view_init(elev=28, azim=215)
    
    # Subplot 2: Biểu đồ cột bóc tách Gradient & Near-Zero Ratio từng tầng
    ax2 = fig.add_subplot(gs[1])
    layers = ['Tầng 1 (Sống)', 'Tầng 2 (CHẾT)', 'Tầng 3 (Sống)', 'Tầng 4 (Sống)']
    grad_norm = [0.425, 0.0008, 0.380, 0.295]
    nzr_ratio = [0.15, 0.94, 0.22, 0.30]
    
    x = np.arange(len(layers))
    width = 0.35
    
    rects1 = ax2.bar(x - width/2, grad_norm, width, label='Độ lớn Gradient |∇L|', color=['#3B82F6', '#EF4444', '#3B82F6', '#3B82F6'])
    rects2 = ax2.bar(x + width/2, nzr_ratio, width, label='Tỷ lệ Gradient ≈ 0 (NZR)', color=['#93C5FD', '#FCA5A5', '#93C5FD', '#93C5FD'], hatch='//')
    
    ax2.axhline(y=0.85, color='#DC2626', linestyle='--', linewidth=1.5, label='Ngưỡng Tầng Chết (dead_nzr = 0.85)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers, fontweight='bold')
    ax2.set_ylabel('Giá trị viễn trắc (Chuẩn hóa)')
    ax2.set_title("Chẩn đoán Vi mô Từng Tầng (Per-Layer Diagnostics)", fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.45)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    hud_text = (
        "CHẨN ĐOÁN VIỄN TRẮC (TELEMETRY):\n"
        "• Tầng 2 NZR: 0.94 > 0.85 (Tầng chết!)\n"
        "• Tầng 1 & 3: Hoạt động mạnh (|∇L| > 0.35)\n"
        "• Bản chất: Góc quay Tầng 2 bị kẹt ma trận đơn vị,\n"
        "  không đóng góp thông tin vào phép đo.\n"
        "• Hành động: SOFT_MASK & PRUNE_LAYER(2)\n"
        "  (Cắt bỏ 24 góc ZYZ và CNOT không tốn Loss!)"
    )
    ax2.text(0.03, 0.96, hud_text, transform=ax2.transAxes, fontsize=8.8, verticalalignment='top',
             fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.4", fc="#1F2937", ec="#374151", alpha=0.9), color="#F9FAFB")
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/03_dead_layer.png", dpi=300)
    plt.close()
    
    # 3.2 Plotly Interactive HTML
    fig_plotly = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'xy'}]],
        subplot_titles=("Cảnh quan cắt lớp: θ⁽¹⁾ (Dốc) vs θ⁽²⁾ (Phẳng lì)", "Gradient & Near-Zero Ratio theo từng tầng")
    )
    fig_plotly.add_trace(
        go.Surface(x=theta1, y=theta2, z=Loss, colorscale='Blues', showscale=False),
        row=1, col=1
    )
    fig_plotly.add_trace(
        go.Bar(x=layers, y=grad_norm, name='Gradient Norm |∇L|', marker_color=['#3B82F6', '#DC2626', '#3B82F6', '#3B82F6']),
        row=1, col=2
    )
    fig_plotly.add_trace(
        go.Bar(x=layers, y=nzr_ratio, name='Near-Zero Ratio (NZR)', marker_color=['#93C5FD', '#FCA5A5', '#93C5FD', '#93C5FD']),
        row=1, col=2
    )
    fig_plotly.add_trace(
        go.Scatter(x=layers, y=[0.85]*len(layers), mode='lines', line=dict(color='red', dash='dash'), name='Ngưỡng chết (0.85)'),
        row=1, col=2
    )
    
    fig_plotly.update_layout(
        title="<b>3. HIỆN TƯỢNG TẦNG CHẾT (DEAD LAYER)</b><br><sup>Gradient tiến về 0 cục bộ ở một tầng đơn lẻ (Các tầng khác vẫn học tốt, cần kích hoạt Pruning)</sup>",
        font=dict(family="Plus Jakarta Sans, sans-serif", size=12),
        barmode='group'
    )
    fig_plotly.write_html(f"{OUT_DIR}/03_dead_layer.html")
    print("   [OK] Đã lưu: figures/diagnostics/03_dead_layer.png & .html")

# ==============================================================================
# 4. BARREN PLATEAU (BÌNH NGUYÊN CẰN CỖI)
# ==============================================================================
def build_barren_plateau():
    print("-> Đang tạo Trực quan hóa 4: BÌNH NGUYÊN CẰN CỖI (Barren Plateau)...")
    
    # 4.1 Matplotlib PNG
    theta1 = np.linspace(-np.pi, np.pi, 200)
    theta2 = np.linspace(-np.pi, np.pi, 200)
    T1, T2 = np.meshgrid(theta1, theta2)
    # Sa mạc phẳng lì tại Loss = 0.693 (ln 2) với độ gợn sóng chỉ 0.0006
    Loss = 0.6931 + 0.0006 * np.sin(2 * T1) * np.cos(2 * T2)
    
    # Viên bi bất động qua 30 bước
    steps = 30
    traj_x = 0.5 + 0.002 * np.random.randn(steps)
    traj_y = -0.4 + 0.002 * np.random.randn(steps)
    traj_loss = 0.6931 + 0.0006 * np.sin(2 * traj_x) * np.cos(2 * traj_y)
    
    fig = plt.figure(figsize=(14, 6), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1.0])
    
    ax1 = fig.add_subplot(gs[0], projection='3d')
    surf = ax1.plot_surface(T1, T2, Loss, cmap='bone', alpha=0.85, edgecolor='none')
    ax1.plot(traj_x, traj_y, traj_loss, color='#EF4444', linewidth=3, label='Adam bất động (∇L = 0)')
    ax1.scatter([traj_x[0]], [traj_y[0]], [traj_loss[0]], color='#DC2626', s=120, marker='o')
    ax1.set_title("Cảnh quan Hàm mất mát 3D (Bình nguyên Cằn cỗi)", fontsize=13, fontweight='bold', pad=15)
    ax1.set_xlabel("Tham số $\\theta_1$", labelpad=8)
    ax1.set_ylabel("Tham số $\\theta_2$", labelpad=8)
    ax1.set_zlabel("Validation Loss", labelpad=8)
    ax1.set_zlim(0.68, 0.71)
    ax1.view_init(elev=25, azim=210)
    
    ax2 = fig.add_subplot(gs[1])
    cs = ax2.contourf(T1, T2, Loss, levels=20, cmap='bone', alpha=0.85)
    plt.colorbar(cs, ax=ax2, label='Validation Loss (Đồng màu ~0.693)')
    ax2.plot(traj_x, traj_y, color='#EF4444', linewidth=2)
    ax2.scatter(traj_x[0], traj_y[0], color='#DC2626', s=180, marker='o', edgecolors='black', label='Điểm khởi tạo ngẫu nhiên')
    
    # Vẽ các mũi tên gradient vi mô
    ax2.text(0.0, 0.0, "SA MẠC PHẲNG LÌ\n|∇L| ~ 10⁻⁵ trên TOÀN BỘ không gian\nThuật toán tối ưu bị 'đóng băng'",
             fontsize=10, fontweight='bold', color='#7F1D1D', ha='center',
             bbox=dict(boxstyle="round,pad=0.4", fc="#FEE2E2", ec="#DC2626"))
    
    hud_text = (
        "CHẨN ĐOÁN VIỄN TRẮC (TELEMETRY):\n"
        "• Loss toàn cục: 0.6931 ≈ ln(2) (Đoán mò)\n"
        "• Phương sai Gradient: Var ~ O(2⁻ⁿ) -> 0\n"
        "• Bản chất: Mạch quá sâu, đạt Haar 2-design,\n"
        "  gradient triệt tiêu TRÊN TOÀN KHÔNG GIAN.\n"
        "• Hành động: EMERGENCY DEPTH REDUCTION\n"
        "  (Cắt giảm tầng khẩn cấp để khôi phục gradient)"
    )
    ax2.text(0.03, 0.96, hud_text, transform=ax2.transAxes, fontsize=9.5, verticalalignment='top',
             fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", fc="#1F2937", ec="#374151", alpha=0.9), color="#F9FAFB")
    
    ax2.set_title("Mặt phẳng Đẳng mức: Không có Dốc", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Tham số $\\theta_1$")
    ax2.set_ylabel("Tham số $\\theta_2$")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/04_barren_plateau.png", dpi=300)
    plt.close()
    
    # 4.2 Plotly Interactive Animated HTML
    frames = []
    for k in range(steps):
        cur_x = traj_x[:k+1]
        cur_y = traj_y[:k+1]
        cur_z = traj_loss[:k+1]
        
        frame = go.Frame(
            data=[
                go.Surface(x=theta1, y=theta2, z=Loss, colorscale='Greys', opacity=0.85, showscale=False),
                go.Scatter3d(x=cur_x, y=cur_y, z=cur_z, mode='lines+markers', line=dict(color='#DC2626', width=5)),
                go.Scatter3d(x=[traj_x[k]], y=[traj_y[k]], z=[traj_loss[k]], mode='markers+text',
                             marker=dict(size=10, color='#DC2626'),
                             text=[f"Epoch {k+1}: Bất động | Loss={traj_loss[k]:.4f} | |∇L|≈0.0000"],
                             textposition="top center")
            ],
            name=f"frame_{k}"
        )
        frames.append(frame)
        
    fig_plotly = go.Figure(
        data=[
            go.Surface(x=theta1, y=theta2, z=Loss, colorscale='Greys', opacity=0.85, colorbar=dict(title="Loss")),
            go.Scatter3d(x=[traj_x[0]], y=[traj_y[0]], z=[traj_loss[0]], mode='lines+markers', line=dict(color='#DC2626', width=5),
                         name='Quỹ đạo bất động'),
            go.Scatter3d(x=[traj_x[0]], y=[traj_y[0]], z=[traj_loss[0]], mode='markers+text',
                         marker=dict(size=10, color='#DC2626'), name='Viên bi',
                         text=[f"Epoch 1: Loss={traj_loss[0]:.4f}"], textposition="top center")
        ],
        layout=go.Layout(
            title="<b>4. HIỆN TƯỢNG BÌNH NGUYÊN CẰN CỖI (BARREN PLATEAU)</b><br><sup>Gradient triệt tiêu trên TOÀN BỘ không gian do tập trung độ đo Haar (Cần cắt giảm tầng khẩn cấp)</sup>",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=12),
            scene=dict(
                xaxis_title="Tham số θ₁", yaxis_title="Tham số θ₂", zaxis_title="Validation Loss",
                zaxis=dict(range=[0.68, 0.71]),
                camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.2))
            ),
            updatemenus=[dict(
                type="buttons", showactive=False, x=0.05, y=0.05,
                buttons=[
                    dict(label="▶ Play Animation", method="animate", args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True)]),
                    dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                ]
            )],
            sliders=[dict(
                steps=[dict(method="animate", args=[[f"frame_{k}"], dict(mode="immediate", frame=dict(duration=0, redraw=True))], label=f"Ep {k+1}") for k in range(steps)],
                x=0.2, y=0.05, len=0.75
            )]
        ),
        frames=frames
    )
    fig_plotly.write_html(f"{OUT_DIR}/04_barren_plateau.html")
    print("   [OK] Đã lưu: figures/diagnostics/04_barren_plateau.png & .html")

if __name__ == "__main__":
    print("\n=== BẮT ĐẦU TẠO 4 BỘ TRỰC QUAN HÓA CHẨN ĐOÁN ĐỘC LẬP ===")
    build_convergence()
    build_local_minimum()
    build_dead_layer()
    build_barren_plateau()
    print("\n=== HOÀN TẤT TOÀN BỘ 4 BỘ HÌNH ẢNH & ANIMATION TẠI: figures/diagnostics/ ===")
