import rasterio
from rasterio import windows


def split_tiff_image(input_tiff, output_tiff_1, output_tiff_2):
    with rasterio.open(input_tiff) as dataset:
        # Calculate the split line (half of the width)
        split_width = dataset.width // 2

        # Define window for the first half
        window_1 = windows.Window(
            col_off=0, row_off=0, width=split_width, height=dataset.height
        )

        # Define window for the second half
        window_2 = windows.Window(
            col_off=split_width,
            row_off=0,
            width=dataset.width - split_width,
            height=dataset.height,
        )

        # Read and write the first half of the image
        kwargs = dataset.meta.copy()
        kwargs.update(
            {
                "width": window_1.width,
                "height": window_1.height,
                "transform": rasterio.windows.transform(window_1, dataset.transform),
            }
        )
        with rasterio.open(output_tiff_1, "w", **kwargs) as dst:
            dst.write(dataset.read(window=window_1))

        # Read and write the second half of the image
        kwargs.update(
            {
                "width": window_2.width,
                "height": window_2.height,
                "transform": rasterio.windows.transform(window_2, dataset.transform),
            }
        )
        with rasterio.open(output_tiff_2, "w", **kwargs) as dst:
            dst.write(dataset.read(window=window_2))
