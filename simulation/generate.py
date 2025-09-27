import pymunk
import numpy as np

def generate_trajectory_data(num_balls=5, num_steps=100, dt=0.1, 
                           box_size=200, max_velocity=50, radius=20,
                           include_velocity=True, add_walls=True):
    """
    Generate collision trajectory data for multi-agent prediction.
    
    Args:
        num_balls: Number of balls to simulate
        num_steps: Number of time steps to simulate
        dt: Time step size
        box_size: Size of the simulation box
        max_velocity: Maximum initial velocity
        radius: Ball radius
        include_velocity: If True, also return velocity data
        add_walls: If True, add walls to bounce off
    
    Returns:
        positions: (num_balls, num_steps, 2) array of positions
        velocities: (num_balls, num_steps, 2) array if include_velocity=True
    """
    space = pymunk.Space()
    space.damping = 0.99  # Slight damping for realism
    
    # Add walls if requested (creates a box)
    if add_walls:
        walls = [
            pymunk.Segment(space.static_body, (-box_size, -box_size), (-box_size, box_size), 1),
            pymunk.Segment(space.static_body, (-box_size, box_size), (box_size, box_size), 1),
            pymunk.Segment(space.static_body, (box_size, box_size), (box_size, -box_size), 1),
            pymunk.Segment(space.static_body, (box_size, -box_size), (-box_size, -box_size), 1)
        ]
        for wall in walls:
            wall.elasticity = 1.0
            space.add(wall)
    
    balls = []
    positions = []
    velocities = []
    
    # Initialize random balls
    for i in range(num_balls):
        # Create body with moment of inertia
        moment = pymunk.moment_for_circle(1, 0, radius)
        body = pymunk.Body(1, moment)
        
        # Random position (ensure no overlap)
        max_attempts = 100
        for _ in range(max_attempts):
            pos = np.random.uniform(-box_size + radius, box_size - radius, 2)
            body.position = tuple(pos)
            
            # Check for overlaps with existing balls
            overlap = False
            for other in balls:
                dist = np.linalg.norm(np.array(body.position) - np.array(other.position))
                if dist < 2 * radius:
                    overlap = True
                    break
            
            if not overlap:
                break
        
        # Random velocity
        body.velocity = tuple(np.random.uniform(-max_velocity, max_velocity, 2))
        
        # Create shape
        shape = pymunk.Circle(body, radius)
        shape.elasticity = 0.95  # Slightly inelastic collisions
        shape.friction = 0.1
        
        space.add(body, shape)
        balls.append(body)
        positions.append([])
        if include_velocity:
            velocities.append([])
    
    # Simulate and record trajectories
    for step in range(num_steps):
        for i, ball in enumerate(balls):
            positions[i].append(list(ball.position))
            if include_velocity:
                velocities[i].append(list(ball.velocity))
        space.step(dt)
    
    positions = np.array(positions)  # Shape: (num_balls, num_steps, 2)
    
    if include_velocity:
        velocities = np.array(velocities)
        return positions, velocities
    
    return positions