import numpy as np
import random
import math
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

random.seed(3)

class CENoiseMap:
    
    class NumberGenerator:
        """Helper class for generating random numbers with specific distribution"""
        
        @staticmethod
        def next(sd: float, mean: float, precision: float = 0.001) -> float:
            """
            Generate a random number using logistic distribution
            
            Args:
                sd: Standard deviation
                mean: Mean value
                precision: Precision threshold (default: 0.001)
            
            Returns:
                Random float value
            """
            x = random.random()
            
            while x < precision or x > 1 - precision:
                x = random.random()
            
            return math.log(x / (1 - x)) / 0.74036268949 * sd + mean
    
    @staticmethod
    def step_map(size: List[int], mean, sd) -> np.ndarray:
        """
        Generate a step map with random values
        
        Args:
            size: List containing [width, height] of the map
            mean: Mean value for random generation
            sd: Standard deviation for random generation
            
        Returns:
            2D numpy array with random values
            
        Raises:
            ValueError: If size doesn't contain exactly two values
        """
        if len(size) != 2:
            raise ValueError("Size must contain exactly two values")
        
        result = np.zeros((size[0], size[1]), dtype=np.float32)

        for i in range(size[0]):
            for j in range(size[1]):
                result[i, j] = CENoiseMap.NumberGenerator.next(
                    sd, mean
                    )
        
        return result
    
    @staticmethod
    def render_map(step_map: np.ndarray, size: List[int], starting_point: List[int], starting_height: float) -> np.ndarray:
        """
        Render the noise map by spreading height values from a starting point
        
        Time Complexity: O(n * m) where n and m are the matrix dimensions
        
        Args:
            step_map: Initial step map as 2D numpy array
            size: List containing [width, height] of the map
            starting_point: List containing [x, y] coordinates of starting point
            starting_height: Initial height value to add at starting point
            
        Returns:
            Rendered 2D numpy array with spread height values
        """

        # Create copies to avoid modifying the original
        step_map = step_map.copy()
        
        # Using matrices to check if nodes have been visited
        # Advantages: access always O(1)
        # Dictionary of HashSets would end up with the same size
        loaded_map = np.zeros((size[0], size[1]), dtype=int)
        
        # Using fixed sized array for storing possible next values
        # Advantages: random access always O(1)
        # Swapping variables: O(1)
        max_positions = size[0] * size[1] + 1
        positions_to_check = np.zeros((max_positions, 2), dtype=int)
        positions_to_check[0, 0] = starting_point[0]
        positions_to_check[0, 1] = starting_point[1]
        list_size = 1
        
        curr = starting_point.copy()

        connections = [
            [(-1,0),(0,1),(1,0),(0,-1)],
            [(-1,0),(0,1),(1,0),(0,-1), (-1,1),(-1,-1),(1,1),(1,-1)],
            [(0,-1), (-1,0), (0,1), (1,0)],
            [ (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0)]
        ]
        
        def is_inside_map(x: int, y: int) -> bool:
            """Check if coordinates are within map boundaries"""
            return not (x < 0 or y < 0 or x >= size[0] or y >= size[1])
        
        step_map[starting_point[0], starting_point[1]] += starting_height

        index = 0
        while list_size > 0:
            # CHOOSING THE NEXT NODE
            # 
            # This justifies using a matrix instead of a list
            # Random access -> order does not matter
            # Complexity of removing item in list = O(n)
            # Complexity of swapping items in array = O(1)

            # Choosing next item
            """
            probs =  {
                "random" : 1/3,
                "stack" : 1/3,
                "gausian-center" : 1/3,
                "gausian-edges" : 1
            }
            kind = random.random()
            for key in probs.keys():
                kind -= probs[key]
                if kind < 0:
                    break
            
            match key:
                case "stack":
                    index = 0
                case "random":
                    index = random.randint(0, list_size-1)
                case "gausian-center":
                    index = int(CENoiseMap.NumberGenerator.next(3, list_size/2) % list_size)
                case "gausian-edges":
                    index = int((CENoiseMap.NumberGenerator.next(6, list_size/2) + list_size/2) % list_size)
            """
            index = list_size - 1
            curr[0] = positions_to_check[index, 0]
            curr[1] = positions_to_check[index, 1]
            
            # "Removing" item from list by swapping with last element
            positions_to_check[index, 0] = positions_to_check[list_size - 1, 0]
            positions_to_check[index, 1] = positions_to_check[list_size - 1, 1]
            list_size -= 1
            #index += 1
            
            # Adding adjacent coordinates to the list
            sum_height = 0.0
            count = 0
            
            
            surroundings = []

            for i, j in connections[3]:
                x = curr[0] + i
                y = curr[1] + j
                if not is_inside_map(x, y):
                    continue
                if loaded_map[x, y] == 0:
                    if (i, j) in connections[2]:
                        surroundings.append((x,y))
                elif loaded_map[x,y] == 2:
                    sum_height += step_map[x, y]
                    count += 1
                    
            random.shuffle(surroundings)

            for (i,j) in surroundings:
                # Adding to the list
                positions_to_check[list_size, 0] = i
                positions_to_check[list_size, 1] = j
                list_size += 1
                            
                # Confirming addition on Matrix
                loaded_map[i, j] = 1
            
            if count > 0:  # Avoid division by zero
                step_map[curr[0], curr[1]] += sum_height / count
            loaded_map[curr[0], curr[1]] = 2
        
        return step_map
    
    @staticmethod
    def render_to_image(noise_map: np.ndarray, 
                       output_path: str = "noise_map.png",
                       style: str = "grayscale",
                       colormap: str = "terrain",
                       size: Optional[Tuple[int, int]] = None) -> Image.Image:
        """
        Render noise map to an image file
        
        Args:
            noise_map: 2D numpy array containing height values
            output_path: Path to save the image (default: "noise_map.png")
            style: Rendering style - "grayscale", "heatmap", "contour", or "terrain"
            colormap: Matplotlib colormap name (for heatmap/terrain styles)
            size: Optional tuple (width, height) to resize the image
            
        Returns:
            PIL Image object
        """
        # Normalize the noise map to 0-1 range
        normalized = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
        
        if style == "grayscale":
            # Convert to grayscale (0-255)
            gray_values = (normalized * 255).astype(np.uint8)
            img = Image.fromarray(gray_values, mode='L')
            
        elif style == "heatmap":
            # Create heatmap using matplotlib colormap
            plt.figure(figsize=(10, 10))
            plt.imshow(normalized, cmap=colormap, interpolation='bilinear')
            plt.colorbar(label='Height')
            plt.title('Noise Map - Heatmap')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Return PIL image for consistency
            img = Image.open(output_path)
            
        elif style == "contour":
            # Create contour plot
            plt.figure(figsize=(10, 10))
            x = np.arange(normalized.shape[1])
            y = np.arange(normalized.shape[0])
            X, Y = np.meshgrid(x, y)
            
            levels = 20  # Number of contour levels
            contour = plt.contourf(X, Y, normalized, levels=levels, cmap=colormap)
            plt.colorbar(contour, label='Height')
            plt.title('Noise Map - Contour')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            img = Image.open(output_path)
            
        elif style == "terrain":
            # Create terrain-like visualization with custom colors
            # Define terrain colormap: deep blue -> blue -> green -> yellow -> brown -> white
            colors = ['#000080', '#0000FF', '#00FF00', "#CDC600", '#999999', '#FFFFFF']
            n_bins = 256
            terrain_cmap = LinearSegmentedColormap.from_list('terrain', colors, N=n_bins)
            
            # Apply colormap
            colored = terrain_cmap(normalized)
            # Convert to RGB (remove alpha channel)
            rgb_array = (colored[:, :, :3] * 255).astype(np.uint8)
            img = Image.fromarray(rgb_array, mode='RGB')
            
        else:
            raise ValueError(f"Unknown style: {style}. Use 'grayscale', 'heatmap', 'contour', or 'terrain'")
        
        # Resize if requested
        if size:
            img = img.resize(size, Image.Resampling.LANCZOS)
        
        # Save the image
        img.save(output_path)
        return img
    
    @staticmethod
    def heightmap_to_colormap(heightmap, show=True, save_path=None):
        """
        Converts a 2D NumPy heightmap into a terrain-colored image using an Earth-like color scale.

        Parameters:
        - heightmap: 2D NumPy array of elevation values
        - show: whether to display the plot with matplotlib
        - save_path: if provided, saves the image to this path (e.g., "map.png")
        
        Returns:
        - The matplotlib figure and axis objects for further customization
        """
        # Define elevation breakpoints (can adjust these based on your heightmap's value range)
        bounds = [-99999, 0, 20, 40, 100, 150, 200, 99999]
        
        # Define the corresponding colors
        colors = [
            "#000080",  # Deep ocean - dark blue
            "#4682B4",  # Shallow ocean - steel blue
            "#D2B48C",  # Beach - tan
            "#228B22",  # Lowlands - forest green
            "#808000",  # Hills - olive
            "#A0522D",  # Mountains - sienna
            "#FFFFFF",  # Snowy peaks - white
        ]
        
        # Create colormap and normalization
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(bounds, cmap.N)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(heightmap, cmap=cmap, norm=norm)
        ax.axis('off')
        
        # Optional: colorbar for reference
        cbar = fig.colorbar(im, ax=ax, shrink=0.7, ticks=bounds, orientation='vertical')
        cbar.set_label('Elevation')

        # Save or show
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        
        return fig, ax

    @staticmethod
    def render_multiple_views(noise_map: np.ndarray, base_filename: str = "noise_map"):
        """
        Render the noise map in multiple styles for comparison
        
        Args:
            noise_map: 2D numpy array containing height values
            base_filename: Base filename (without extension) for output files
        """
        styles = {
            "grayscale": {"colormap": None},
            "heatmap": {"colormap": "viridis"},
            "contour": {"colormap": "plasma"},
            "terrain": {"colormap": None}
        }
        
        images = {}
        for style, params in styles.items():
            filename = f"{base_filename}_{style}.png"
            if params["colormap"]:
                img = CENoiseMap.render_to_image(noise_map, filename, style, params["colormap"])
            else:
                img = CENoiseMap.render_to_image(noise_map, filename, style)
            images[style] = img
            print(f"Saved {style} version: {filename}")
        
        return images
    
    @staticmethod
    def create_animated_generation(size: List[int], mean: float, sd: float, 
                                 starting_point: List[int], starting_height: float,
                                 output_path: str = "generation_steps"):
        """
        Create a series of images showing the step-by-step generation process
        
        Args:
            size: Map dimensions
            mean: Mean for random generation
            sd: Standard deviation for random generation
            starting_point: Starting coordinates
            starting_height: Initial height
            output_path: Base path for output images
        """
        # This would require modifying the render_map function to yield intermediate steps
        # For now, we'll create a simplified version showing before/after
        
        initial_map = CENoiseMap.step_map(size, mean, sd)
        rendered_map = CENoiseMap.render_map(initial_map, size, starting_point, starting_height)
        
        # Save initial state
        CENoiseMap.render_to_image(initial_map, f"{output_path}_initial.png", "terrain")
        
        # Save final state
        CENoiseMap.render_to_image(rendered_map, f"{output_path}_final.png", "terrain")
        
        print(f"Saved generation steps: {output_path}_initial.png and {output_path}_final.png")

    @staticmethod
    def generate_map(size_x, size_y, mean = 0, sd = 1, sh = 20) -> None:
        size = [size_x, size_y]

        # Generate initial step map
        initial_map = CENoiseMap.step_map(size, mean, sd)
        
        # Render the map starting from center
        starting_point = [random.randint(0, size[0]), random.randint(0, size[1])]
        
        return CENoiseMap.render_map(initial_map, size, [0,0], sh)

    @staticmethod
    def image_from_1d_matrix(noise_map: np.ndarray, final_size: List[int]):
        image_matrix: np.ndarray = np.zeros((final_size[0], final_size[1]), dtype=float)

        for col in range(final_size[1]):
            for i in range(int(noise_map[0,col])):
                try:
                    image_matrix[col, -1 - i] = 1
                except:
                    pass 
        return image_matrix
    
    @staticmethod
    def create_animated_generation(size: List[int], mean: float, sd: float, 
                                 starting_point: List[int], starting_height: float,
                                 output_path: str = "generation_steps"):
        """
        Create a series of images showing the step-by-step generation process
        
        Args:
            size: Map dimensions
            mean: Mean for random generation
            sd: Standard deviation for random generation
            starting_point: Starting coordinates
            starting_height: Initial height
            output_path: Base path for output images
        """
        # This would require modifying the render_map function to yield intermediate steps
        # For now, we'll create a simplified version showing before/after
        
        initial_map = CENoiseMap.step_map(size, mean, sd)
        rendered_map = CENoiseMap.render_map(initial_map, size, starting_point, starting_height)
        
        # Save initial state
        CENoiseMap.render_to_image(initial_map, f"{output_path}_initial.png", "terrain")
        
        # Save final state
        CENoiseMap.render_to_image(rendered_map, f"{output_path}_final.png", "terrain")
        
        print(f"Saved generation steps: {output_path}_initial.png and {output_path}_final.png")

    @staticmethod
    def oneD(n, start, mean = 0, sd = 1, sh = 20):
        noise = [
            CENoiseMap.NumberGenerator.next
            (
                sd = (lambda x: (8))(i), 
                mean = (lambda x: (0))(i)
            )
            for i in range(n)
            ]
        
        
        sample_altitudes = noise.copy()
        sample_altitudes[start] += sh

        for i in range(start, n):
            sample_altitudes[i] = sample_altitudes[i-1] + noise[i]
        for i in range(0, start, -1):
            sample_altitudes[i] = sample_altitudes[i+1] + noise[i]

        return sample_altitudes
    
    @staticmethod
    def step_list(size: int, mean: float, sd: float):
        
        result = np.zeros((size), dtype=np.float32)
        
        for i in range(size):
            result[i] = CENoiseMap.NumberGenerator.next(sd, mean)
        
        return result

    @staticmethod
    def render_1D_map(step_map: np.ndarray, size: int, starting_point: int, starting_height: float):
        """
        Render the noise map by spreading height values from a starting point
        
        Time Complexity: O(n * m) where n and m are the matrix dimensions
        
        Args:
            step_map: Initial step map as 2D numpy array
            size: List containing [width, height] of the map
            starting_point: List containing [x, y] coordinates of starting point
            starting_height: Initial height value to add at starting point
            
        Returns:
            Rendered 2D numpy array with spread height values
        """

        # Create copies to avoid modifying the original
        step_map = step_map.copy()
        
        # Using matrices to check if nodes have been visited
        # Advantages: access always O(1)
        # Dictionary of HashSets would end up with the same size
        render_map = np.zeros((size,1), dtype=int)
        
        # Using fixed sized array for storing possible next values
        # Advantages: random access always O(1)
        # Swapping variables: O(1)
        max_positions = size + 1
        positions_to_check = np.zeros((max_positions,1), dtype=int)
        positions_to_check[0,0] = starting_point
        list_size = 1
        
        curr = starting_point
        
        def is_inside_map(x: int) -> bool:
            """Check if coordinates are within map boundaries"""
            return not (x < 0 or x >= size)
        
        step_map[starting_point,0] += starting_height

        index = 0
        while list_size > 0:
            # CHOOSING THE NEXT NODE
            # 
            # This justifies using a matrix instead of a list
            # Random access -> order does not matter
            # Complexity of removing item in list = O(n)
            # Complexity of swapping items in array = O(1)
            
            # Choosing next item
            #index = random.randint(0, list_size) if random.random() < 0.0 else list_size -1

            #index = random.randint(0,list_size)
            index = 0
            curr = positions_to_check[index,0]
            
            # "Removing" item from list by swapping with last element
            positions_to_check[index,0] = positions_to_check[list_size - 1]
            list_size -= 1
            #index += 1
            
            # Adding adjacent coordinates to the list
            sum_height = 0.0
            count = 0
            
            surroundings = []
            for i in [-1,1]:
                p = curr + i
                if is_inside_map(i):
                    if render_map[p,0] == 2:  # Node already visited
                        # Adds to the average adjacent height
                        sum_height += step_map[p,0]
                        count += 1
                    elif render_map[p,0] == 0:  # Node not yet visited
                        # Adding to the list
                        surroundings.append(p)
                                                    
            random.shuffle(surroundings)

            for (i) in surroundings:
                positions_to_check[list_size, 0] = i
                list_size += 1
                render_map[i,0] = 1
            
            if count > 0:  # Avoid division by zero
                step_map[curr, 0] += sum_height / count
                render_map[i, 0] = 2
        
        return step_map

    @staticmethod
    def render_map_linearly(size, mean, sd, sh, sp):
        # Create Helper lists
        x_helper = CENoiseMap.oneD(size[0], sp[0], mean = mean, sd = sd, sh = sh)
        y_helper = CENoiseMap.oneD(size[1], sp[1], mean = mean, sd = sd, sh = sh)

        gaussian_noise = CENoiseMap.step_map(size, mean, sd)
        # Average the helper list coodinates + fluctuation

        noise_map = gaussian_noise.copy()
        for i in range(size[0]):
            for j in range(size[1]):
                noise_map[i, j] += (x_helper[i] + y_helper[j]) / 2
            
        return noise_map
    
# Example usage:
if __name__ == "__main__":
    import math
    size = [256, 256]
    mean = lambda x, y: -1
    sd = lambda x, y: 0.1
    # Generate initial step map
    initial_map = CENoiseMap.step_map(size, mean, sd)
    
    # Render the map starting from center
    starting_point = [int(size[0]/2), int(size[1]/2)]#[random.randint(0,size[0]), random.randint(0,size[1])]
    #
    starting_height = 150
    

    rendered_map = CENoiseMap.render_map(initial_map, size, starting_point, starting_height)
    CENoiseMap.heightmap_to_colormap(rendered_map, show=False, save_path="z_gpt.png")
    
    print(f"Generated noise map with shape: {rendered_map.shape}")
    print(f"Height range: {rendered_map.min():.3f} to {rendered_map.max():.3f}")
    
    # Render to different image formats
    print("\nRendering images...")
    
    # Single image with terrain style
    terrain_img = CENoiseMap.render_to_image(rendered_map, "terrain_map.png", "terrain")
    print("Created terrain_map.png")
    
    # Grayscale version
    grayscale_img = CENoiseMap.render_to_image(rendered_map, "grayscale_map.png", "grayscale")
    print("Created grayscale_map.png")

    grayscale_img = CENoiseMap.render_to_image(initial_map, "grayscale_map_ste_map.png", "grayscale")
    print("Created grayscale_step_map.png")

    """
    # Heatmap version
    heatmap_img = CENoiseMap.render_to_image(rendered_map, "heatmap_map.png", "heatmap", "hot")
    print("Created heatmap_map.png")
    
    # Create all views at once
    print("\nCreating multiple views...")
    CENoiseMap.render_multiple_views(rendered_map, "complete_noise_map")
    
    # Show generation steps
    print("\nCreating generation comparison...")
    CENoiseMap.create_animated_generation(size, mean, sd, starting_point, starting_height)
    
    print("\nAll images have been generated successfully!")


    import topo_map

    topo_image = topo_map.noise_to_topological_map(rendered_map)
    topo_image.save("topological_map.png")
    print("Topological map saved as topological_map.png")
    
    # Create heightmap
    heightmap = topo_map.create_heightmap_image(rendered_map, colormap='terrain')
    heightmap.save("heightmap.png")
    print("Heightmap saved as heightmap.png")
    
    # Create comparison image
    comparison = topo_map.save_comparison_image(rendered_map)
    
    # Display image info
    print(f"Topological map size: {topo_image.size}")
    print(f"Heightmap size: {heightmap.size}")
    """