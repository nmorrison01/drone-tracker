import torch

def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps:0") 
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    return device


# Functions to plot 3D trajectory
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go

def plot_trajectory_static(trajectory_path):

    # load text file
    trajectory = np.loadtxt(trajectory_path)

    # split into x, y, z
    x, y, z = trajectory[:,0], trajectory[:,1], trajectory[:,2]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x, y, z, color="red", label="Trajectory")
    ax.scatter(x[0], y[0], z[0], c="green", s=100, label="Start")
    ax.scatter(x[-1], y[-1], z[-1], c="blue", s=100, label="End")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Drone Trajectory")
    ax.legend()

    plt.show()


def plot_trajectory_static_colored(trajectory_path):

    # load trajectory from txt file
    trajectory = np.loadtxt(trajectory_path)
    x, y, z = trajectory[:,0], trajectory[:,1], trajectory[:,2]

    # create a color array based on time (index)
    t = np.linspace(0, 1, len(x))  # normalize to [0,1]
    colors = cm.viridis(t) 

    # plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # scatter for per-point coloring
    sc = ax.scatter(x, y, z, c=t, cmap="viridis", marker="o", s=30)

    # connect points with line
    ax.plot(x, y, z, color="gray", alpha=0.5, linewidth=1)

    # mark start and end
    ax.scatter(x[0], y[0], z[0], c="green", s=100, label="Start")
    ax.scatter(x[-1], y[-1], z[-1], c="red", s=100, label="End")

    # labels and colorbar
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Drone Trajectory with Time Coloring")
    fig.colorbar(sc, ax=ax, label="Normalized Time (0=start, 1=end)")
    ax.legend()

    plt.show()

def plot_trajectory_interactive(trajectory_path):
    
    # load trajectory from txt file
    trajectory = np.loadtxt(trajectory_path)
    x, y, z = trajectory[:,0], trajectory[:,1], trajectory[:,2]

    # create color scale based on time
    t = np.linspace(0, 1, len(x))

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers+lines',
        marker=dict(
            size=5,
            color=t,
            colorscale='Viridis',
            showscale=True
        ),
        line=dict(
            color='gray',
            width=2
        )
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title="Interactive 3D Drone Trajectory"
    )

    fig.show()