import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

print("Loading NPZ...")
data = np.load("outputs/fk_pos.npz")
global_pos = data['pos']
connections = data['conn']

num_frames = len(global_pos)

print("Rendering GIF...")
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

min_b = np.min(global_pos, axis=(0, 1))
max_b = np.max(global_pos, axis=(0, 1))
max_range = np.array([max_b[0]-min_b[0], max_b[1]-min_b[1], max_b[2]-min_b[2]]).max() / 2.0
mid_x = (max_b[0]+min_b[0]) * 0.5
mid_y = (max_b[1]+min_b[1]) * 0.5
mid_z = (max_b[2]+min_b[2]) * 0.5

lines = []
for _ in range(len(connections)):
    line, = ax.plot([], [], [], 'bo-', linewidth=2, markersize=2)
    lines.append(line)

def update(frame):
    if frame % 50 == 0:
        print(f"Frame {frame}")
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_title(f"Retargeted Skeleton - Frame {frame}")
    
    pts = global_pos[frame]
    for idx, (p, c) in enumerate(connections):
        p1 = pts[p]
        p2 = pts[c]
        lines[idx].set_data([p1[0], p2[0]], [p1[1], p2[1]])
        lines[idx].set_3d_properties([p1[2], p2[2]])
        
    return lines

ani = animation.FuncAnimation(fig, update, frames=min(num_frames, 300), interval=33, blit=False)
output_gif = "outputs/retargeted_skeleton.gif"
ani.save(output_gif, writer='pillow')
print(f"Saved GIF to {output_gif}")
