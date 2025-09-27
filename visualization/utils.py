import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from IPython.display import HTML
import matplotlib.colors as mcolors

def animate_trajectories(trajectories, box_size=200, interval=50, 
                         trail_length=10, show_trails=True, 
                         ball_radius=20, figsize=(8, 8)):
    """
    Animate ball trajectories in a Jupyter notebook.
    
    Args:
        trajectories: numpy array of shape (num_balls, num_steps, 2)
        box_size: size of the boundary box
        interval: milliseconds between frames
        trail_length: number of previous positions to show as trail
        show_trails: whether to show trajectory trails
        ball_radius: radius of balls for visualization
        figsize: figure size tuple
    
    Returns:
        HTML animation object for Jupyter display
    """
    num_balls, num_steps, _ = trajectories.shape
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-box_size - 20, box_size + 20)
    ax.set_ylim(-box_size - 20, box_size + 20)
    ax.set_aspect('equal')
    ax.set_facecolor('#f0f0f0')
    
    # Draw boundaries
    boundary_style = dict(color='black', linewidth=2)
    ax.plot([-box_size, -box_size], [-box_size, box_size], **boundary_style)
    ax.plot([-box_size, box_size], [box_size, box_size], **boundary_style)
    ax.plot([box_size, box_size], [box_size, -box_size], **boundary_style)
    ax.plot([box_size, -box_size], [-box_size, -box_size], **boundary_style)
    
    # Generate colors for each ball
    colors = plt.cm.tab10(np.linspace(0, 1, num_balls))
    
    # Initialize ball artists
    ball_circles = []
    trail_lines = []
    
    for i in range(num_balls):
        # Create circle for each ball
        circle = Circle((0, 0), ball_radius, 
                       color=colors[i], 
                       ec='black', 
                       linewidth=1.5,
                       alpha=0.8,
                       zorder=10)
        ax.add_patch(circle)
        ball_circles.append(circle)
        
        # Create trail line
        if show_trails:
            line, = ax.plot([], [], 
                          color=colors[i], 
                          alpha=0.3, 
                          linewidth=2,
                          zorder=5)
            trail_lines.append(line)
    
    # Add title
    title = ax.text(0, box_size + 30, '', 
                   ha='center', 
                   fontsize=12,
                   fontweight='bold')
    
    def init():
        """Initialize animation."""
        for circle in ball_circles:
            circle.center = (0, 0)
        for line in trail_lines:
            line.set_data([], [])
        title.set_text('Frame 0')
        return ball_circles + trail_lines + [title]
    
    def animate(frame):
        """Update animation frame."""
        # Update ball positions
        for i, circle in enumerate(ball_circles):
            circle.center = trajectories[i, frame]
        
        # Update trails
        if show_trails:
            start_frame = max(0, frame - trail_length)
            for i, line in enumerate(trail_lines):
                trail_x = trajectories[i, start_frame:frame+1, 0]
                trail_y = trajectories[i, start_frame:frame+1, 1]
                line.set_data(trail_x, trail_y)
        
        # Update title
        title.set_text(f'Frame {frame}/{num_steps-1} | Time: {frame*0.1:.1f}s')
        
        return ball_circles + trail_lines + [title]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=num_steps, interval=interval,
                                  blit=True, repeat=True)
    
    plt.close(fig)  # Prevent static display
    return HTML(anim.to_jshtml())