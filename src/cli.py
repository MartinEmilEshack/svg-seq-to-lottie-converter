#!/usr/bin/env python3
"""
SVG/XML to Lottie CLI Tool

Usage:
    python cli.py input.svg output.json
    python cli.py input.xml output.json              # XML with SVG content (e.g., Fabric.js export)
    python cli.py input.svg output.json --optimize
    python cli.py input.svg  # outputs to input.json in same directory
    python cli.py frames.zip output.json  # zip containing SVG/XML frames
"""

import argparse
import json
import sys
import os
from pathlib import Path
import copy
import tempfile
import zipfile
from typing import List, Tuple, Optional

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.svg import convert_svg_to_lottie, convert_svg_to_lottie_def
import cairosvg
from tempfile import NamedTemporaryFile


def is_svg_content(filename: str) -> bool:
    """
    Check if file contains valid SVG content.
    Works for both .svg and .xml files that contain SVG structure.
    """
    import xml.etree.cElementTree as et
    tag = None
    try:
        with open(filename, "r") as f:
            for _, el in et.iterparse(f, ('start',)):
                tag = el.tag
                break
    except (et.ParseError, FileNotFoundError):
        pass
    return tag == '{http://www.w3.org/2000/svg}svg'


def is_svg(filename: str) -> bool:
    """Check if file is a valid SVG (alias for backward compatibility)."""
    return is_svg_content(filename)


def _convert_svg_to_lottie_dict(
    input_file: str,
    optimize: bool = False,
    embed_images: bool = False,
    frame_rate: Optional[float] = None
) -> dict:
    """
    Convert SVG/XML file to Lottie JSON without writing to disk.
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if not is_svg_content(str(input_path)):
        raise ValueError(f"Invalid SVG/XML file (no SVG content found): {input_file}")

    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    has_images = '<image' in content or 'xlink:href' in content

    tmp_path = None
    try:
        with NamedTemporaryFile(delete=False, suffix=".svg") as tmp:
            tmp_path = tmp.name
        cairosvg.svg2svg(file_obj=open(input_path, 'rb'), write_to=tmp_path)

        if optimize:
            result = convert_svg_to_lottie(tmp_path, embed_images=embed_images, frame_rate=frame_rate)
        else:
            result = convert_svg_to_lottie_def(tmp_path, embed_images=embed_images, frame_rate=frame_rate)

        if isinstance(result, dict) and 'error!' in result:
            raise RuntimeError(f"Conversion failed: {result.get('error!', 'Unknown error')}")

        if has_images:
            image_result = _extract_images_from_svg(str(input_path), embed_images, content)
            if image_result:
                if 'assets' not in result or result['assets'] is None:
                    result['assets'] = []
                result['assets'].extend(image_result.get('assets', []))

                image_layers = list(reversed(image_result.get('layers', [])))
                shape_layers = result['layers']

                merged_layers = _merge_layers_by_position(
                    str(input_path),
                    tmp_path,
                    image_layers,
                    shape_layers
                )

                if merged_layers:
                    result['layers'] = merged_layers
                else:
                    print("Warning: Position matching failed, using simple layer order")
                    background_images = [l for l in image_layers if l.get('refId', '') in ['image_0', 'image_1']]
                    foreground_images = [l for l in image_layers if l.get('refId', '') not in ['image_0', 'image_1']]
                    result['layers'] = foreground_images + shape_layers + background_images

        return result
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _prefix_asset_ids(assets: list, prefix: str) -> tuple[list, dict]:
    updated_assets = []
    id_mapping = {}
    for asset in assets:
        if isinstance(asset, dict) and asset.get('id'):
            new_id = f"{prefix}_{asset['id']}"
            id_mapping[asset['id']] = new_id
            asset = {**asset, 'id': new_id}
        updated_assets.append(asset)
    return updated_assets, id_mapping


def _merge_zip_frames(frames: List[dict]) -> dict:
    if not frames:
        raise ValueError("No SVG frames found in zip archive.")

    base = copy.deepcopy(frames[0])
    total_frames = len(frames)
    combined_layers = []
    combined_assets = []
    layer_index = 1

    for frame_index, frame in enumerate(frames):
        assets = frame.get('assets') or []
        updated_assets, id_mapping = _prefix_asset_ids(assets, f"frame_{frame_index}")
        combined_assets.extend(updated_assets)

        for layer in frame.get('layers', []):
            layer_copy = copy.deepcopy(layer)
            layer_copy['ip'] = frame_index
            layer_copy['op'] = frame_index + 1
            layer_copy['ind'] = layer_index
            if 'refId' in layer_copy and layer_copy['refId'] in id_mapping:
                layer_copy['refId'] = id_mapping[layer_copy['refId']]
            combined_layers.append(layer_copy)
            layer_index += 1

    base['ip'] = 0
    base['op'] = total_frames
    base['layers'] = combined_layers
    base['assets'] = combined_assets
    return base


def _extract_zip_svg_entries(zip_path: Path) -> List[Tuple[str, bytes]]:
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Input zip is not a valid zip file: {zip_path}")

    entries = []
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        for info in zip_file.infolist():
            if info.is_dir():
                continue
            filename = info.filename
            suffix = Path(filename).suffix.lower()
            if suffix not in {'.svg', '.xml'}:
                continue
            entries.append((filename, zip_file.read(info)))

    entries.sort(key=lambda entry: entry[0])
    return entries


def convert_zip(
    input_file: str,
    output_json: str,
    optimize: bool = False,
    pretty: bool = True,
    embed_images: bool = False,
    frame_rate: Optional[float] = None
) -> dict:
    """
    Convert ZIP of SVG/XML files to a single multi-frame Lottie JSON.
    """
    input_path = Path(input_file)
    output_path = Path(output_json)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    entries = _extract_zip_svg_entries(input_path)
    if not entries:
        raise ValueError(f"No SVG/XML files found in zip: {input_file}")

    frame_results = []
    base_dimensions = None
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        for filename, data in entries:
            sanitized_name = Path(filename).name
            extracted_path = temp_path / sanitized_name
            extracted_path.write_bytes(data)

            result = _convert_svg_to_lottie_dict(
                str(extracted_path),
                optimize=optimize,
                embed_images=embed_images,
                frame_rate=frame_rate
            )

            frame_dims = (result.get('w'), result.get('h'))
            if base_dimensions is None:
                base_dimensions = frame_dims
            elif frame_dims != base_dimensions:
                raise ValueError(
                    f"Frame dimensions mismatch in zip. Expected {base_dimensions}, got {frame_dims} for {filename}."
                )
            frame_results.append(result)

    combined_result = _merge_zip_frames(frame_results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        if pretty:
            json.dump(combined_result, f, indent=2, ensure_ascii=False)
        else:
            json.dump(combined_result, f, ensure_ascii=False)

    return combined_result


def convert(
    input_file: str,
    output_json: str,
    optimize: bool = False,
    pretty: bool = True,
    embed_images: bool = False,
    frame_rate: Optional[float] = None
) -> dict:
    """
    Convert SVG/XML file to Lottie JSON.
    
    Args:
        input_file: Path to input SVG or XML file (must contain SVG content)
        output_json: Path to output JSON file
        optimize: Use optimized conversion mode
        pretty: Pretty print JSON output
        embed_images: Download and embed images as base64 (default: keep URL references)
    
    Returns:
        Conversion result dictionary
    """
    input_path = Path(input_file)
    output_path = Path(output_json)

    result = _convert_svg_to_lottie_dict(
        str(input_path),
        optimize=optimize,
        embed_images=embed_images,
        frame_rate=frame_rate
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        if pretty:
            json.dump(result, f, indent=2, ensure_ascii=False)
        else:
            json.dump(result, f, ensure_ascii=False)

    return result


def _extract_images_from_svg(svg_path: str, embed_images: bool = False, content: str = None) -> dict:
    """
    Extract only image elements from SVG file.
    Returns a dict with assets and layers for images only.
    """
    from core.svg.convert import Parser
    from xml.etree import ElementTree
    
    try:
        parser = Parser(embed_images=embed_images)
        tree = ElementTree.parse(svg_path)
        
        # Create a minimal animation just to extract images
        animation = parser.parse_etree(tree)
        
        # Return only image-related data
        result = {
            'assets': animation.assets if animation.assets else [],
            'layers': []
        }
        
        # Extract only image layers (ty=2)
        for layer in animation.layers:
            # Pydantic v1 uses .dict(), v2 uses .model_dump()
            if hasattr(layer, 'model_dump'):
                layer_dict = layer.model_dump(exclude_none=True, by_alias=True)
            else:
                layer_dict = layer.dict(exclude_none=True, by_alias=True)
            if layer_dict.get('ty') == 2:
                result['layers'].append(layer_dict)
        
        return result
    except Exception as e:
        print(f"Warning: Failed to extract images: {e}")
        import traceback
        traceback.print_exc()
        return None


def _merge_layers_by_position(original_svg_path: str, cairo_svg_path: str, 
                               image_layers: list, shape_layers: list) -> list:
    """
    Merge image and shape layers preserving the exact order from original SVG.
    
    This function:
    1. Analyzes original SVG to get the order of elements (image/text groups)
    2. Extracts Y position of each character from CairoSVG output  
    3. Groups shape layers by Y position to match original text groups
    4. Interleaves image and text layers in the correct order
    
    Returns merged layer list, or None if matching fails.
    """
    import xml.etree.ElementTree as ET
    import re
    
    try:
        # Step 1: Parse original SVG to get element order and text bounding boxes (X and Y)
        original_tree = ET.parse(original_svg_path)
        original_root = original_tree.getroot()
        
        # Collect element order: list of ('image', image_index) or ('text', bbox)
        element_order = []
        image_index = 0
        text_bboxes = []  # [(min_x, max_x, min_y, max_y, element_index), ...]
        
        for child in original_root:
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if tag == 'g':
                transform = child.get('transform', '')
                match = re.search(r'matrix\(([^)]+)\)', transform)
                
                has_text = child.find('.//{http://www.w3.org/2000/svg}text') is not None
                has_image = child.find('.//{http://www.w3.org/2000/svg}image') is not None
                
                if has_image:
                    element_order.append(('image', image_index))
                    image_index += 1
                elif has_text and match:
                    parts = list(map(float, match.group(1).replace(',', ' ').split()))
                    if len(parts) >= 6:
                        # matrix(a, b, c, d, tx, ty) where a=scaleX, d=scaleY
                        scale_x, scale_y = parts[0], parts[3]
                        tx, ty = parts[4], parts[5]
                        
                        # Get X and Y positions of all tspans
                        tspans = child.findall('.//{http://www.w3.org/2000/svg}tspan')
                        x_positions = set()
                        y_positions = set()
                        for t in tspans:
                            tspan_x = float(t.get('x', 0))
                            tspan_y = float(t.get('y', 0))
                            screen_x = tx + tspan_x * scale_x
                            screen_y = ty + tspan_y * scale_y
                            x_positions.add(round(screen_x))
                            y_positions.add(round(screen_y))
                        
                        if x_positions and y_positions:
                            elem_idx = len(element_order)
                            bbox = (min(x_positions), max(x_positions), 
                                   min(y_positions), max(y_positions))
                            element_order.append(('text', bbox))
                            text_bboxes.append((bbox[0], bbox[1], bbox[2], bbox[3], elem_idx))
        
        # Step 2: Parse CairoSVG output to get character X and Y positions
        cairo_tree = ET.parse(cairo_svg_path)
        cairo_root = cairo_tree.getroot()
        
        char_positions = []  # [(shape_layer_index, x, y), ...]
        shape_idx = 0
        
        for child in cairo_root:
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if tag == 'g':
                for use_elem in child.iter():
                    use_tag = use_elem.tag.split('}')[-1]
                    if use_tag == 'use':
                        x = float(use_elem.get('x', 0))
                        y = float(use_elem.get('y', 0))
                        char_positions.append((shape_idx, round(x), round(y)))
                        shape_idx += 1
                        break
        
        # Note: char_positions may differ from shape_layers count
        # (e.g., SVG with shapes but no text, or text + other shapes)
        # This is normal and doesn't affect the layer ordering
        
        # Step 3: Group shape layers by matching X and Y position to text bounding boxes
        text_group_layers = {}  # element_index -> list of shape layers
        
        for shape_idx, x_pos, y_pos in char_positions:
            if shape_idx >= len(shape_layers):
                break
            
            # Find which text group this character belongs to
            matched = False
            for min_x, max_x, min_y, max_y, elem_idx in text_bboxes:
                # Allow some tolerance (±10 pixels for X, ±5 pixels for Y)
                # X tolerance is larger because characters can extend beyond tspan x values
                if (min_x - 50 <= x_pos <= max_x + 50) and (min_y - 5 <= y_pos <= max_y + 5):
                    if elem_idx not in text_group_layers:
                        text_group_layers[elem_idx] = []
                    text_group_layers[elem_idx].append(shape_layers[shape_idx])
                    matched = True
                    break
            
            if not matched:
                # Character doesn't match any group, add to a catch-all
                if -1 not in text_group_layers:
                    text_group_layers[-1] = []
                text_group_layers[-1].append(shape_layers[shape_idx])
        
        # Step 4: Build layer groups in original SVG order (bottom to top)
        # Each group is a list of layers (image is single-element list, text is multi-element)
        groups_in_svg_order = []
        
        for idx, (elem_type, value) in enumerate(element_order):
            if elem_type == 'image':
                # Add image layer as a single-element group
                if value < len(image_layers):
                    groups_in_svg_order.append([image_layers[value]])
            else:  # 'text'
                # idx is the element index in element_order, get its shape layers
                if idx in text_group_layers:
                    groups_in_svg_order.append(text_group_layers[idx])
        
        # Add any unmatched characters as a group
        if -1 in text_group_layers:
            groups_in_svg_order.append(text_group_layers[-1])
        
        # Step 5: Reverse group order (Lottie: first = top, SVG: last = top)
        # But keep the internal order within each group
        groups_in_lottie_order = list(reversed(groups_in_svg_order))
        
        # Flatten the groups into final layer list
        merged_layers = []
        for group in groups_in_lottie_order:
            merged_layers.extend(group)
        
        return merged_layers
        
    except Exception as e:
        print(f"Warning: Position-based matching failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Convert SVG/XML files to Lottie JSON format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python cli.py input.svg output.json
  python cli.py input.xml output.json              # XML with SVG content
  python cli.py input.svg output.json --optimize
  python cli.py input.svg                          # outputs to input.json
  python cli.py input.svg -o /path/to/output.json
  python cli.py input.svg --compact                # minified JSON
  python cli.py frames.zip --compact               # zip containing SVG/XML frames
        '''
    )
    
    parser.add_argument(
        'input',
        help='Input SVG/XML file path or a zip containing SVG/XML frames'
    )
    
    parser.add_argument(
        'output',
        nargs='?',
        default=None,
        help='Output JSON file path (default: same as input with .json extension)'
    )
    
    parser.add_argument(
        '-o', '--output-file',
        dest='output_file',
        help='Output JSON file path (alternative to positional argument)'
    )
    
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Use optimized conversion mode'
    )
    
    parser.add_argument(
        '--compact',
        action='store_true',
        help='Output compact JSON (no indentation)'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress output messages'
    )
    
    parser.add_argument(
        '--embed-images',
        action='store_true',
        help='Download and embed images as base64 (default: keep URL references)'
    )

    parser.add_argument(
        '--frame-rate',
        type=float,
        default=30,
        help='Frame rate for the output animation (default: 30)'
    )
    
    args = parser.parse_args()
    
    # Determine output path
    output_path = args.output_file or args.output
    if output_path is None:
        # Default: same name as input but with .json extension
        input_path = Path(args.input)
        output_path = str(input_path.with_suffix('.json'))
    
    try:
        if not args.quiet:
            print(f"Converting: {args.input}")
            print(f"Output to:  {output_path}")
            if args.optimize:
                print("Mode:       optimized")

        input_path = Path(args.input)
        is_zip_input = input_path.suffix.lower() == '.zip' or zipfile.is_zipfile(input_path)
        if is_zip_input:
            result = convert_zip(
                args.input,
                output_path,
                optimize=args.optimize,
                pretty=not args.compact,
                embed_images=args.embed_images,
                frame_rate=args.frame_rate
            )
        else:
            result = convert(
                args.input,
                output_path,
                optimize=args.optimize,
                pretty=not args.compact,
                embed_images=args.embed_images,
                frame_rate=args.frame_rate
            )
        
        if not args.quiet:
            # Get some stats
            layers_count = len(result.get('layers', []))
            width = result.get('w', 0)
            height = result.get('h', 0)
            file_size = os.path.getsize(output_path)
            
            print(f"\n✅ Conversion successful!")
            print(f"   Dimensions: {width}x{height}")
            print(f"   Layers:     {layers_count}")
            print(f"   File size:  {file_size:,} bytes")
        
        return 0
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    
    except ValueError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    
    except RuntimeError as e:
        print(f"❌ Conversion error: {e}", file=sys.stderr)
        return 2
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
