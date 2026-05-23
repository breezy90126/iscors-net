import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import random
from matplotlib.animation import FuncAnimation

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def verify_data(config):
    data_dir = config['train_folder']
    results_dir = config['results_dir']
    os.makedirs(results_dir, exist_ok=True)

    # 1. 尋找現有的影片與軌跡檔案
    all_files = os.listdir(data_dir)
    movie_files = sorted([f for f in all_files if f.startswith('movie_') and f.endswith('.tif')])
    traj_files = sorted([f for f in all_files if f.startswith('movie_') and f.endswith('_traj.npy')])

    if not movie_files:
        print(f"Error: No generated data found in {data_dir}. Please run `python main.py` first.")
        return

    # 2. 隨機選一個樣本
    idx = random.randint(0, len(movie_files) - 1)
    movie_path = os.path.join(data_dir, movie_files[idx])
    traj_path = os.path.join(data_dir, traj_files[idx])

    print(f"Verifying sample: {movie_files[idx]}")

    # 讀取數據
    import tifffile
    movie = tifffile.imread(movie_path) # [T, H, W]
    trajectory = np.load(traj_path)      # [T, N, 3]
    
    T, H, W = movie.shape
    _, N_particles, _ = trajectory.shape

    # 3. 繪製軌跡圖 (XY Plane)
    plt.figure(figsize=(8, 8))
    colors = plt.cm.jet(np.linspace(0, 1, N_particles))
    
    for p in range(N_particles):
        x = trajectory[:, p, 0]
        y = trajectory[:, p, 1]
        plt.plot(x, y, '-', color=colors[p], alpha=0.3) # 連線
        plt.scatter(x, y, s=10, color=colors[p], label=f'P{p+1}') # 點
        
        # 計算位移
        dist = np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2)
        print(f"Particle {p+1} total displacement: {dist*1e9:.2f} nm")

    plt.title(f"Particle Trajectories (XY Plane) - Sample {idx}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(results_dir, 'verification_trajectories.png'))
    plt.close()

    # 4. 生成動畫 GIF (每幀 0.4s)
    print("Generating verification GIF (0.4s per frame)...")
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 歸一化以方便顯示
    v_min, v_max = movie.min(), movie.max()
    im = ax.imshow(movie[0], cmap='gray', vmin=v_min, vmax=v_max)
    ax.set_title(f"Frame 0 / {T}")
    ax.axis('off')

    def update(frame_idx):
        im.set_data(movie[frame_idx])
        ax.set_title(f"Frame {frame_idx} / {T}")
        return [im]

    # interval = 400 ms (0.4s)
    ani = FuncAnimation(fig, update, frames=T, interval=400, blit=True)
    gif_path = os.path.join(results_dir, 'verification_movie.gif')
    ani.save(gif_path, writer='pillow')
    plt.close()

    print(f"Verification complete!")
    print(f"- Trajectory plot saved to: {os.path.join(results_dir, 'verification_trajectories.png')}")
    print(f"- Animated GIF saved to: {gif_path}")

if __name__ == "__main__":
    if os.path.exists('params.json'):
        config = load_config('params.json')
        verify_data(config)
    else:
        print("params.json not found")
