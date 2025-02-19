import argparse
import os
import json
import sys
import numpy as np
import logging
from datetime import datetime
import cv2

from panorai.pipeline.pipeline import ProjectionPipeline
from panorai.pipeline.pipeline_data import PipelineData
from projection import ProjectionRegistry
from panorai.sampler.registry import SamplerRegistry
from panorai.blender.registry import BlenderRegistry

def setup_logging(verbose):
    logging_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging_level,
    )

def list_all_projections_and_samplers():
    """
    Always log which projections and samplers exist, for user awareness.
    """
    # Projections
    projs = ProjectionRegistry.list_projections()
    logging.info("Available Projections are:")
    for p in projs:
        logging.info(f"  - {p}")

    # Samplers
    sams = SamplerRegistry.list_samplers()
    logging.info("Available Samplers are:")
    for s in sams:
        logging.info(f"  - {s}")

    # Blenders
    blens = BlenderRegistry.list_blenders()
    logging.info("Available Blenders are:")
    for s in blens:
        logging.info(f"  - {s}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Panorai CLI for spherical image processing.",
        epilog="""
Examples:

1) Gnomonic projection with shadow_angle=30
   panorai-cli --input ../images/sample2.npz --kwargs shadow_angle=30 --array_files rgb z

2) Gnomonic projection with shadow_angle=30, rotate latitude=5 and longitude=5
   panorai-cli --input ../images/sample2.npz --kwargs shadow_angle=30 delta_lat=5 delta_lon=5 --array_files rgb z

3) IcosahedronSampler with shadow_angle=30, subdivisions=2
   panorai-cli --sampler_name=IcosahedronSampler --input ../images/sample2.npz --kwargs shadow_angle=30 subdivisions=2 --array_files rgb z

4) FibonacciSampler with n_points=30
   panorai-cli --sampler_name=FibonacciSampler --input ../images/sample2.npz --kwargs n_points=30 --array_files rgb z
"""
    )

    # If user runs with no flags at all, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # Utility options
    parser.add_argument("--list-projections", action="store_true", help="List all available projections and exit.")
    parser.add_argument("--list-samplers", action="store_true", help="List all available samplers and exit.")
    parser.add_argument("--list-files", action="store_true", help="List all files inside the provided NPZ input.")
    parser.add_argument("--show-pipeline", action="store_true", help="Show details of the instantiated pipeline object.")

    # Input parameters
    parser.add_argument("--input", type=str, help="Path to the input file or directory.")
    parser.add_argument("--array_files", type=str, nargs="*", help="Keys for data in the .npz file (e.g., rgb, depth).")

    # Projection parameters
    parser.add_argument("--projection_name", type=str, default="gnomonic",
                        help="Name of the projection to use (default='gnomonic').")
    parser.add_argument("--sampler_name", type=str, default="CubeSampler",
                        help="Name of the sampler to use (default='CubeSampler').")
    parser.add_argument("--blender_name", type=str, default=None,
                        help="Name of the blender to use (default='ClosestBlender).")
    parser.add_argument("--operation", choices=["project", "backward"], help="Operation to perform.")
    parser.add_argument("--kwargs", nargs="*", default=[], help="Additional arguments in key=value format.")
    parser.add_argument("--fov_deg", type=float, default=90., help="Field of view for the gnomonic projection - 90 is the max.")
    parser.add_argument("--method", type=str, default="ndimage", help="Interpolation package either ndimage or cv2 are accepted")
    

    # Rotation parameters (newly added)
    parser.add_argument("--delta_lat", type=float, default=0., help="Latitude rotation in degrees.")
    parser.add_argument("--delta_lon", type=float, default=0., help="Longitude rotation in degrees.")

    # Output options
    parser.add_argument("--output_dir", type=str, default=".cache",
                        help="Base directory to save the output files (default='.cache').")
    parser.add_argument("--save_npz", action="store_true", default=True,
                        help="Save results as a single .npz file (default=True). Disable to save storage.")
    parser.add_argument("--save_png", action="store_true", default=True,
                        help="Save illustrative PNG images (default=True). Disable to save storage.")
    parser.add_argument("--cmap", type=str, default="jet",
                        help="Colormap name for single-channel arrays (default='jet').")

    # Logging and verbosity
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Enable verbose logging (default=True). Pass --no-verbose to silence (see below).")
    parser.add_argument("--no-verbose", action="store_true", help="Disable verbose logging (override).")

    return parser.parse_args()

def create_unique_output_dir(base_dir, args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params = f"{args.projection_name}_{args.operation or 'both'}"
    unique_dir = os.path.join(base_dir, f"run_{timestamp}_{params}")
    os.makedirs(unique_dir, exist_ok=True)
    return unique_dir

def save_metadata(output_dir, args):
    metadata = {
        "input": args.input,
        "projection_name": args.projection_name,
        "sampler_name": args.sampler_name,
        "blender_name": args.blender_name,
        "operation": args.operation,
        "kwargs": args.kwargs,
        "output_dir": output_dir,
        "command": " ".join(sys.argv),
        "save_npz": args.save_npz,
        "save_png": args.save_png,
        "verbose": not args.no_verbose,
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"Saved metadata to {metadata_path}.")

def list_npz_files(input_path):
    with np.load(input_path) as data:
        logging.info("Files in NPZ:")
        for key in data.keys():
            logging.info(f" - {key}")

###############################################################################
# Utility code: normalization, colormap, flattening, etc.
###############################################################################
def normalize_array(array: np.ndarray) -> np.ndarray:
    if array.size == 0:
        return array.astype(np.uint8)
    min_val, max_val = np.min(array), np.max(array)
    if min_val == max_val:
        return np.zeros_like(array, dtype=np.uint8)
    norm = (array - min_val) / (max_val - min_val)
    return (norm * 255).astype(np.uint8)

def apply_colormap(array_2d: np.ndarray, cmap_name: str = "jet") -> np.ndarray:
    norm_img = normalize_array(array_2d)
    cmap_dict = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "viridis": cv2.COLORMAP_VIRIDIS,
    }
    cv2_cmap = cmap_dict.get(cmap_name.lower(), cv2.COLORMAP_JET)
    color_img = cv2.applyColorMap(norm_img, cv2_cmap)
    return color_img

def compose_3channel(array_3d: np.ndarray) -> np.ndarray:
    out = np.zeros_like(array_3d, dtype=np.uint8)
    for c in range(3):
        out[..., c] = normalize_array(array_3d[..., c])
    return out

def _flatten_result_for_npz(result_dict, prefix=""):
    """
    Recursively traverse and collect all np.ndarray objects
    into a dict { <flattened_key>: (ndarray, original_subkey) }.
    """
    flat_data = {}
    for key, val in result_dict.items():
        new_key = f"{prefix}.{key}" if prefix else key

        if isinstance(val, dict):
            nested = _flatten_result_for_npz(val, prefix=new_key)
            flat_data.update(nested)
            continue

        if hasattr(val, "__dict__") and all(isinstance(v, np.ndarray) for v in val.__dict__.values()):
            for subk, arr in val.__dict__.items():
                final_key = f"{new_key}.{subk}"
                flat_data[final_key] = (arr, subk)
            continue

        if isinstance(val, np.ndarray):
            flat_data[new_key] = (val, key)
        else:
            logging.debug(f"Ignoring key '{new_key}' - not dict or ndarray.")
    return flat_data

###############################################################################
# Main saving logic
###############################################################################
def save_output(
    combined_result: dict,
    output_dir: str,
    save_npz: bool,
    operation: str = None,
    save_png: bool = False,
    cmap: str = "jet"
):
    """
    - combined_result is a dictionary that may contain:
      * 'project': <dict returned by pipeline.project(...)>
      * 'backward': <dict returned by pipeline.backward(...)>
      or both, if no operation was specified.

    - We first flatten and save the raw arrays into .npz (if save_npz=True).
    - Then we optionally create PNG images from the same arrays, applying
      colormaps or normalizations as needed for visualization (without altering
      the raw data that went into the .npz).
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Flatten original raw arrays BEFORE any normalization
    flat_dict = _flatten_result_for_npz(combined_result)

    # 2) Save the NPZ with raw data
    if save_npz:
        npz_path = os.path.join(output_dir, "output.npz")
        arrays_for_npz = {k: v[0] for k, v in flat_dict.items()}  # The original arrays
        np.savez_compressed(npz_path, **arrays_for_npz)
        logging.info(f"Saved all arrays to {npz_path}.")
        logging.info("Note: The .npz file contains the most complete data. Use --no-save_npz to save storage.")

    # 3) Optionally create PNG images for quick inspection
    if save_png:
        logging.info("Saving illustrative .png files (one per array if possible). Use --no-save_png to save storage.")
        for full_key, (array, original_name) in flat_dict.items():
            safe_key = full_key.replace(".", "_")

            # If recognized as "rgb", handle color conversion
            if original_name.lower() == "rgb":
                if array.ndim == 3 and array.shape[2] == 3:
                    # Ensure 8-bit
                    if array.dtype != np.uint8:
                        array = normalize_array(array)
                    # Convert from RGB -> BGR
                    bgr_img = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
                    png_path = os.path.join(output_dir, f"{safe_key}.png")
                    cv2.imwrite(png_path, bgr_img)
                    logging.info(f"Saved RGB array '{original_name}' to {png_path} (converted to BGR).")
                else:
                    logging.warning(
                        f"'{original_name}' labeled as RGB but shape={array.shape}. Skipping."
                    )
                continue

            # Otherwise numeric data
            if array.ndim == 2:
                color_img = apply_colormap(array, cmap_name=cmap)
                png_path = os.path.join(output_dir, f"{safe_key}_colormap.png")
                cv2.imwrite(png_path, color_img)
                logging.debug(f"Saved single-channel numeric array '{original_name}' to {png_path}.")

            elif array.ndim == 3 and array.shape[2] == 1:
                color_img = apply_colormap(array[..., 0], cmap_name=cmap)
                png_path = os.path.join(output_dir, f"{safe_key}_colormap.png")
                cv2.imwrite(png_path, color_img)
                logging.debug(f"Saved single-channel numeric array '{original_name}' to {png_path}.")

            elif array.ndim == 3 and array.shape[2] == 3:
                composed = compose_3channel(array)
                png_path = os.path.join(output_dir, f"{safe_key}.png")
                cv2.imwrite(png_path, composed)
                logging.debug(f"Saved 3-channel numeric array '{original_name}' as single image {png_path}.")

            else:
                logging.debug(
                    f"Skipping PNG for '{full_key}' (original='{original_name}'), shape {array.shape}"
                    " - not 2D or 3D with 1/3 channels."
                )
    else:
        logging.debug("PNG saving is disabled. Use --save_png to enable it.")

###############################################################################
# Input loading logic
###############################################################################
def load_input(input_path, array_files, preprocess_params):
    if input_path and input_path.endswith(".npz"):
        with np.load(input_path) as data:
            available_keys = list(data.keys())
            if not array_files:
                logging.info("No --array_files specified. Using all available files in the NPZ:")
                for key in available_keys:
                    logging.info(f" - {key}")
                array_files = available_keys
            else:
                missing_keys = [key for key in array_files if key not in available_keys]
                if missing_keys:
                    logging.error("The following keys are not available in the NPZ file:")
                    for key in missing_keys:
                        logging.error(f" - {key}")
                    logging.error("Available keys are:")
                    for key in available_keys:
                        logging.error(f" - {key}")
                    sys.exit(1)
            pipeline_data = PipelineData.from_dict({key: data[key] for key in array_files})
    elif input_path:
        # e.g. .png or .jpg
        from skimage.io import imread
        pipeline_data = PipelineData(rgb=imread(input_path))
    else:
        # no input -> no data
        logging.warning("No input provided. Creating empty PipelineData.")
        pipeline_data = PipelineData(rgb=np.zeros((1,1,3), dtype=np.uint8))

    shadow_angle = preprocess_params.get("shadow_angle", 0)


    if shadow_angle:
        pipeline_data.preprocess(shadow_angle=shadow_angle)

    return pipeline_data

def parse_kwargs(kwargs_list):
    kwargs = {}
    for item in kwargs_list:
        key, value = item.split("=", 1)
        try:
            value = eval(value)
        except (NameError, SyntaxError):
            pass
        kwargs[key] = value
    return kwargs

###############################################################################
# Main
###############################################################################
def main():
    args = parse_args()

    # If user specifically did --no-verbose, override
    if args.no_verbose:
        args.verbose = False

    setup_logging(args.verbose)

    # Always show what's available
    list_all_projections_and_samplers()

    # If user asked only to list projections or samplers, we exit
    if args.list_projections or args.list_samplers or args.list_blenders:
        sys.exit(0)

    # If user wants to list files in an NPZ
    if args.list_files:
        if not args.input or not args.input.endswith(".npz"):
            logging.error("Please provide a valid NPZ input to list files.")
            sys.exit(1)
        list_npz_files(args.input)
        sys.exit(0)

    # If user wants to show pipeline and exit
    if args.show_pipeline:
        pipeline = ProjectionPipeline(
            projection_name=args.projection_name,
            sampler_name=args.sampler_name,
            blender_name=args.blender_name
        )
        logging.info("Instantiated Pipeline Object:")
        logging.info(repr(pipeline))
        sys.exit(0)

    # Inform about default saving
    logging.info("Note: .npz saving is on by default (use --no-save_npz to disable).")
    logging.info("Note: .png saving is on by default for illustration (use --no-save_png to disable).")

    # Parse custom kwargs
    print(args)
    _kwargs = parse_kwargs(args.kwargs)
    kwargs = vars(args)
    kwargs.update(_kwargs)

    delta_lat = kwargs.get("delta_lat", 0)
    delta_lon = kwargs.get("delta_lon", 0)

    print(kwargs)

    if (delta_lat != 0) | (delta_lon != 0):
        rotations = [(delta_lat, delta_lon)]
        print(f'*****rotations: {rotations}')
    else:
        rotations = []
        print(f'*****rotations: {rotations} , - {delta_lat}, {delta_lon}')
    kwargs['rotations'] = rotations

    preprocess_params = {
        "shadow_angle": kwargs.pop("shadow_angle", 0),
    }

    # Create output dir
    output_dir = create_unique_output_dir(args.output_dir, args)
    save_metadata(output_dir, args)

    # Load data
    input_data = load_input(args.input, args.array_files, preprocess_params)

    # Create pipeline
    pipeline = ProjectionPipeline(
        projection_name=args.projection_name,
        sampler_name=args.sampler_name,
        blender_name=args.blender_name
    )

    # Collect results in a single dictionary
    combined_result = {}

    if args.operation == "project":
        # For "project" only => store all points
        combined_result["project"] = pipeline.project(data=input_data, **kwargs)

    elif args.operation == "backward":
        # For "backward" only => store the backward arrays
        combined_result["backward"] = pipeline.backward(data=input_data.as_dict(), **kwargs)

    else:
        # No operation => do both: keep all points + backward
        project_result = pipeline.project(data=input_data, **kwargs)
        backward_result = pipeline.backward(data=project_result, **kwargs)
        combined_result["project"] = project_result
        combined_result["backward"] = backward_result

    # Now save everything into one NPZ (and optionally PNG)
    save_output(
        combined_result,
        output_dir=output_dir,
        save_npz=args.save_npz,
        operation=args.operation,
        save_png=args.save_png,
        cmap=args.cmap
    )

    logging.debug(f"Setup '{pipeline.__repr__}'.")
    logging.debug(f"Projector Setup '{pipeline.projector.config}'.")

if __name__ == "__main__":
    main()