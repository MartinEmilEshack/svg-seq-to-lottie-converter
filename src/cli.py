#!/usr/bin/env python3
"""
SVG to Lottie CLI Tool

Usage:
    python cli.py input.svg output.json
    python cli.py input.svg output.json --optimize
    python cli.py input.svg  # outputs to input.json in same directory
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.svg import convert_svg_to_lottie, convert_svg_to_lottie_def
import cairosvg
from tempfile import NamedTemporaryFile


def is_svg(filename: str) -> bool:
    """Check if file is a valid SVG."""
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


def convert(input_svg: str, output_json: str, optimize: bool = False, pretty: bool = True) -> dict:
    """
    Convert SVG file to Lottie JSON.
    
    Args:
        input_svg: Path to input SVG file
        output_json: Path to output JSON file
        optimize: Use optimized conversion mode
        pretty: Pretty print JSON output
    
    Returns:
        Conversion result dictionary
    """
    input_path = Path(input_svg)
    output_path = Path(output_json)
    
    # Validate input file
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_svg}")
    
    if not is_svg(str(input_path)):
        raise ValueError(f"Invalid SVG file: {input_svg}")
    
    # Preprocess SVG with CairoSVG (normalize SVG)
    with NamedTemporaryFile(delete=False, suffix=".svg") as tmp:
        tmp_path = tmp.name
    
    try:
        cairosvg.svg2svg(file_obj=open(input_path, 'rb'), write_to=tmp_path)
        
        # Convert to Lottie
        if optimize:
            result = convert_svg_to_lottie(tmp_path)
        else:
            result = convert_svg_to_lottie_def(tmp_path)
        
        # Check for conversion errors
        if isinstance(result, dict) and 'error!' in result:
            raise RuntimeError(f"Conversion failed: {result.get('error!', 'Unknown error')}")
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(result, f, indent=2, ensure_ascii=False)
            else:
                json.dump(result, f, ensure_ascii=False)
        
        return result
    
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(
        description='Convert SVG files to Lottie JSON format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python cli.py input.svg output.json
  python cli.py input.svg output.json --optimize
  python cli.py input.svg                          # outputs to input.json
  python cli.py input.svg -o /path/to/output.json
  python cli.py input.svg --compact                # minified JSON
        '''
    )
    
    parser.add_argument(
        'input',
        help='Input SVG file path'
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
        
        result = convert(
            args.input,
            output_path,
            optimize=args.optimize,
            pretty=not args.compact
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
