"""Adapt exported vector labels and panels for dark backgrounds, not sky images.

Usage: python dark_svg.py input.svg output.svg
The input is a pdftocairo SVG export. Embedded rasters and transparency masks
are preserved; only vector fill/stroke colours are changed.
"""

import colorsys
import re
import sys
from xml.etree import ElementTree as ET


def dark_colour(value, attribute):
    """Keep hues while moving dark ink and pale panels to a dark-theme palette."""
    match = re.fullmatch(r"rgb\(([-\d.% ,]+)\)", value)
    if not match:
        return value
    rgb = [float(v.strip('% ')) / (100 if '%' in v else 255)
           for v in match[1].split(',')]
    if min(rgb) > 0.99:
        return '#242833' if attribute == 'fill' else '#ffffff'
    if max(rgb) < 0.01:
        return '#e7eaf0'
    hue, lightness, saturation = colorsys.rgb_to_hls(*rgb)
    colour = colorsys.hls_to_rgb(hue, max(0.18, min(0.85, 1 - lightness)), saturation)
    return '#' + ''.join(f'{round(channel * 255):02x}' for channel in colour)


def convert(source, destination):
    """Write a dark SVG without changing geometry or embedded image data."""
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    tree = ET.parse(source)
    root = tree.getroot()
    protected = {child for node in root.iter()
                 if node.tag.rsplit('}', 1)[-1] in ('mask', 'filter')
                 for child in node.iter()}
    images_before = [dict(node.attrib) for node in root.iter()
                     if node.tag.endswith('}image')]
    root.set('fill', '#e7eaf0')
    for node in root.iter():
        if node in protected:
            continue
        for attribute in ('fill', 'stroke'):
            if attribute in node.attrib:
                node.set(attribute, dark_colour(node.get(attribute), attribute))
    assert images_before == [dict(node.attrib) for node in root.iter()
                             if node.tag.endswith('}image')]
    tree.write(destination, encoding='utf-8', xml_declaration=True)


if __name__ == '__main__':
    assert dark_colour('rgb(0%, 0%, 0%)', 'fill') == '#e7eaf0'
    assert dark_colour('rgb(100%, 100%, 100%)', 'fill') == '#242833'
    assert dark_colour('rgb(100%, 100%, 100%)', 'stroke') == '#ffffff'
    assert dark_colour('none', 'fill') == 'none'
    convert(*sys.argv[1:])
