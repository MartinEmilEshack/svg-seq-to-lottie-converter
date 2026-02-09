# SVG to Lottie Conversion Overview

This project converts SVG files into Lottie JSON by parsing the SVG document into an internal, Pydantic-backed object model and then serializing that model to Lottie. The core conversion logic lives in `src/core/svg/convert.py` and is shared by both the CLI and FastAPI entry points.

## How conversion works

1. **Parse the SVG into an element tree**
   - The converter reads the SVG with Python's `xml.etree.ElementTree`, registers namespaces, and walks the SVG root element.
   - It prefers `viewBox` for sizing (falling back to `width`/`height`) and handles common CSS-style units like `px`, `pt`, `cm`, `%`, and viewport units (`vw`, `vh`, `vmin`, `vmax`).

2. **Build Lottie layers and shapes**
   - Each SVG element is mapped to a Lottie shape or layer (for example, `<rect>` → `Rect`, `<path>` → `Path`).
   - Groups (`<g>` and `<symbol>`) become Lottie groups with inherited transforms, opacity, and visibility rules.
   - Styles are collected from presentation attributes and inline `style` strings, then applied to Lottie `Fill`, `Stroke`, and transform properties.

3. **Apply transforms and styling**
   - The converter supports `transform` attributes such as `translate`, `scale`, `rotate`, `skewX`, `skewY`, and `matrix`.
   - Opacity, stroke width, line caps/joins, and fill/stroke colors (including gradients) are applied as Lottie properties.

4. **Handle gradients, images, and animation timing**
   - Linear and radial gradients are parsed from `<linearGradient>` and `<radialGradient>` and translated to Lottie gradient fills/strokes.
   - `<image>` elements are converted into Lottie image assets and image layers. External images can be embedded as base64 when requested.
   - Simple SVG `<animate>` elements are interpreted to generate Lottie keyframes for supported attributes.

## Supported SVG features

### Elements
- `<svg>` root sizing via `viewBox`, `width`, and `height`.
- Shapes: `<path>`, `<rect>`, `<circle>`, `<ellipse>`, `<line>`, `<polyline>`, `<polygon>`.
- Grouping: `<g>`, `<symbol>`, `<use>` (references by ID).
- Text: `<text>` and nested `<tspan>` (text handling is present but font rendering is disabled by default).
- Images: `<image>` (external URLs or embedded data).
- Gradients: `<linearGradient>`, `<radialGradient>`, and `<stop>`.
- Definitions: `<defs>` for reusable content.

### Styling & attributes
- Presentation attributes and inline `style` declarations for:
  - `fill`, `fill-opacity`
  - `stroke`, `stroke-opacity`, `stroke-width`
  - `stroke-linecap`, `stroke-linejoin`, `stroke-miterlimit`
  - `opacity`, `display`, `visibility`
- Transform attributes: `translate`, `scale`, `rotate`, `skewX`, `skewY`, `matrix`.
- Units: `px`, `em`, `ex`, `in`, `cm`, `mm`, `pt`, `pc`, `%`, `vw`, `vh`, `vmin`, `vmax`, and `Q`.

### Animation support
- Basic `<animate>` elements are converted into Lottie keyframes for compatible attributes on supported shapes (e.g., position/size for ellipses and rectangles).

## Notes and limitations
- Text rendering depends on a font subsystem that is disabled by default in this project, so `<text>` elements are currently parsed for structure and styling but may not render without enabling font support.
- Unsupported or unrecognized SVG features are ignored during parsing.
