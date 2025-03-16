import matplotlib.pyplot as plt
import numpy as np
import os
import math

# Create output directory
output_dir = os.path.join("results", "simple_animation")
frames_dir = os.path.join(output_dir, "frames")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)

def create_crystal_growth_frames(n_frames=100, save_dir=None):
    """
    Create frames of crystal growth simulation.
    
    Args:
        n_frames: Number of frames to generate
        save_dir: Directory to save frames (if None, will use default path)
    """
    # Default save directory
    if save_dir is None:
        save_dir = frames_dir
    
    # Define domain
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    
    # Create spatial grid
    resolution = 50
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Time domain
    t_min, t_max = 0, 0.5
    times = np.linspace(t_min, t_max, n_frames)
    
    # Function to compute crystal interface
    def crystal_interface(x, t):
        # Base position
        base_height = 0.2
        
        # Small curvature in the interface (hump in the middle)
        curvature = 0.05 * np.sin(np.pi * x)
        
        # Growth rate - crystal grows upward with time
        growth_rate = 0.1 * t
        
        return base_height + curvature + growth_rate
    
    # Simplified simulation (since we don't have a trained model)
    def simulate_flow(X, Y, t):
        # Compute interface
        interface_y = crystal_interface(X[0, :], t)
        
        # Create mask for points below the interface
        mask = Y < np.tile(interface_y, (Y.shape[0], 1))
        
        # Simplified velocity field (circular flow)
        cx, cy = 0.5, 0.6  # Center of circulation
        u = -(Y - cy) * (1 + 0.5 * np.sin(2*np.pi*t))  # x-velocity
        v = (X - cx) * (1 + 0.5 * np.sin(2*np.pi*t))   # y-velocity
        
        # Add a bit of randomness to make it look more realistic
        u += 0.05 * np.random.randn(*u.shape)
        v += 0.05 * np.random.randn(*v.shape)
        
        # Apply mask
        u_masked = np.ma.array(u, mask=mask)
        v_masked = np.ma.array(v, mask=mask)
        
        # Pressure field (simplified)
        p = 0.5 - np.sqrt((X - cx)**2 + (Y - cy)**2)
        
        # Temperature field (gradient from hot bottom to cold top)
        T = 1.0 - (Y - y_min) / (y_max - y_min)
        T_masked = np.ma.array(T, mask=mask)
        
        return u_masked, v_masked, p, T_masked, interface_y
    
    # Generate and save each frame
    frame_paths = []
    
    print(f"Generating {n_frames} frames...")
    for frame in range(n_frames):
        # Create a new figure for each frame
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Get current time
        t = times[frame]
        
        # Simulate for this time
        u_masked, v_masked, p, T_masked, interface_y = simulate_flow(X, Y, t)
        
        # Calculate velocity magnitude
        vel_magnitude = np.sqrt(np.maximum(0, u_masked**2 + v_masked**2))
        
        # Plot velocity field
        im1 = axs[0].pcolormesh(X, Y, vel_magnitude, cmap='viridis', shading='auto')
        plt.colorbar(im1, ax=axs[0], label='Velocity magnitude')
        
        # Try to add streamlines (simple version)
        try:
            axs[0].streamplot(x, y, u_masked.T, v_masked.T, color='white', density=1.0)
        except:
            # Fall back to quiver plot if streamplot fails
            stride = 4
            axs[0].quiver(X[::stride, ::stride], Y[::stride, ::stride],
                          u_masked[::stride, ::stride], v_masked[::stride, ::stride],
                          color='white', scale=30, width=0.001)
        
        # Plot crystal interface for velocity plot
        axs[0].plot(x, interface_y, 'k-', linewidth=2)
        axs[0].fill_between(x, interface_y, np.min(y), color='gray', alpha=0.5)
        
        # Pressure field
        im2 = axs[1].pcolormesh(X, Y, p, cmap='RdBu_r', shading='auto')
        plt.colorbar(im2, ax=axs[1], label='Pressure')
        
        # Plot crystal interface for pressure plot
        axs[1].plot(x, interface_y, 'k-', linewidth=2)
        axs[1].fill_between(x, interface_y, np.min(y), color='gray', alpha=0.5)
        
        # Temperature field
        im3 = axs[2].pcolormesh(X, Y, T_masked, cmap='hot', shading='auto')
        plt.colorbar(im3, ax=axs[2], label='Temperature')
        
        # Plot crystal interface for temperature plot
        axs[2].plot(x, interface_y, 'k-', linewidth=2)
        axs[2].fill_between(x, interface_y, np.min(y), color='gray', alpha=0.5)
        
        # Set titles and labels
        titles = ['Velocity Field', 'Pressure Field', 'Temperature Field']
        for i, title in enumerate(titles):
            axs[i].set_xlabel('x')
            axs[i].set_ylabel('y')
            axs[i].set_title(f'{title} at t={t:.2f}')
        
        # Tight layout
        plt.tight_layout()
        
        # Save the frame
        frame_path = os.path.join(save_dir, f"frame_{frame:04d}.png")
        plt.savefig(frame_path, dpi=150)
        plt.close(fig)
        
        frame_paths.append(frame_path)
        
        # Print progress
        if (frame + 1) % 10 == 0 or frame == 0 or frame == n_frames - 1:
            print(f"Generated frame {frame+1}/{n_frames} (t={t:.2f})")
    
    print(f"All {n_frames} frames have been saved to {save_dir}")
    
    # Create a simple HTML file to view frames as slideshow
    html_path = os.path.join(output_dir, "slideshow.html")
    create_slideshow_html(frame_paths, html_path)
    
    return frame_paths, html_path

def create_slideshow_html(frame_paths, output_path):
    """Create a simple HTML slideshow to view frames."""
    frame_names = [os.path.basename(path) for path in frame_paths]
    
    # Create HTML content with string concatenation
    img_tags = ""
    for i, name in enumerate(frame_names):
        img_tags += f'<img src="frames/{name}" id="frame{i}" class="frame">'
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Crystal Growth Simulation</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; text-align: center; }
            .slideshow { max-width: 90%; margin: 0 auto; }
            .slideshow img { max-width: 100%; display: none; }
            .controls { margin: 15px 0; }
            button { padding: 8px 16px; margin: 0 5px; }
            .frame-info { margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>Crystal Growth Simulation</h1>
        <div class="slideshow">
            <!-- Images will be added here -->
    """
    
    html_content += img_tags
    
    # Add frame count without using f-string
    frame_count_html = """
        </div>
        <div class="frame-info">
            Frame: <span id="frameNumber">1</span> / """
    frame_count_html += str(len(frame_paths))
    frame_count_html += """
        </div>
        <div class="controls">
            <button onclick="prevFrame()">Previous</button>
            <button onclick="playPause()">Play/Pause</button>
            <button onclick="nextFrame()">Next</button>
            <br>
            <label>Speed: <input type="range" min="1" max="30" value="10" id="speed"></label>
        </div>

        <script>
            const frames = document.querySelectorAll('.frame');
            let currentFrame = 0;
            let isPlaying = false;
            let playInterval;
            const frameNumber = document.getElementById('frameNumber');
            const speedControl = document.getElementById('speed');

            // Show first frame initially
            showFrame(0);

            function showFrame(n) {
                // Hide all frames
                frames.forEach(frame => frame.style.display = 'none');
                
                // Show the selected frame
                currentFrame = (n + frames.length) % frames.length;
                frames[currentFrame].style.display = 'block';
                frameNumber.textContent = currentFrame + 1;
            }

            function nextFrame() {
                showFrame(currentFrame + 1);
            }

            function prevFrame() {
                showFrame(currentFrame - 1);
            }

            function playPause() {
                if (isPlaying) {
                    clearInterval(playInterval);
                    isPlaying = false;
                } else {
                    isPlaying = true;
                    playInterval = setInterval(() => {
                        nextFrame();
                    }, 1000 / speedControl.value);
                }
            }

            // Update interval when speed changes
            speedControl.addEventListener('input', () => {
                if (isPlaying) {
                    clearInterval(playInterval);
                    playInterval = setInterval(() => {
                        nextFrame();
                    }, 1000 / speedControl.value);
                }
            });
        </script>
    </body>
    </html>
    """
    
    html_content += frame_count_html
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Created slideshow HTML: {output_path}")
    return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create crystal growth simulation frames")
    parser.add_argument('--frames', type=int, default=50, help='Number of frames to generate')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for frames')
    args = parser.parse_args()
    
    # Create the frames
    frames, html_path = create_crystal_growth_frames(
        n_frames=args.frames,
        save_dir=args.output_dir
    )
    
    print(f"\nSimulation complete!")
    print(f"Generated {len(frames)} frames")
    print(f"To view the slideshow, open: {html_path}") 