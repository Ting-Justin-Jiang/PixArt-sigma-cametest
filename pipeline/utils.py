from PIL import Image

def save_grid(images, num_rows, output_path):
    num_images = len(images)
    num_cols = (num_images + num_rows - 1) // num_rows

    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images)

    grid_width = num_cols * max_width
    grid_height = num_rows * max_height

    grid_image = Image.new('RGB', (grid_width, grid_height))

    for index, image in enumerate(images):
        row = index // num_cols
        col = index % num_cols
        x = col * max_width
        y = row * max_height
        grid_image.paste(image, (x, y))

    grid_image.save(output_path)