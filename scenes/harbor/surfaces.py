# SPDX-License-Identifier: CC0-1.0
"""Original baked surface maps and analytical sky. No input photographs."""

import math
from pathlib import Path
import struct
import zlib

import bpy
import numpy as np


# Encode original RGBA arrays directly: each PNG scanline starts with filter 0,
# and chunks carry big-endian lengths plus CRCs. No source photographs are loaded.
def png(path, pixels):
    pixels = np.asarray(np.clip(pixels * 255 + .5, 0, 255), dtype=np.uint8)
    height, width, channels = pixels.shape
    assert channels == 4 and path.parent.is_dir()
    rows = np.zeros((height, 1 + width * 4), dtype=np.uint8)
    rows[:, 1:] = pixels.reshape(height, width * 4)

    # PNG chunk checksums cover both the type tag and compressed/uncompressed data.
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">2I5B", width, height, 8, 6, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(rows.tobytes(), 7)) + chunk(b"IEND", b""))


# Bake bounded-resolution material maps once during authoring. A fixed seed gives
# repeatable grain/pitting choices; rendering later only reads the exported pixels.
def make_maps(directory):
    directory.mkdir(exist_ok=True)
    n = 1024
    v, u = np.mgrid[0:n, 0:n].astype(np.float32) / n
    rng = np.random.default_rng(630117)
    fine = rng.random((n, n), dtype=np.float32) - .5
    grain = .5 * np.sin(2 * math.pi * (u * 61 + .4 * np.sin(v * 17) + .23 * np.sin(v * 41)))
    grain += .25 * np.sin(2 * math.pi * (u * 157 + .6 * np.sin(v * 13)))
    cloud = .33 * np.sin(u * 17 + np.sin(v * 13)) * np.cos(v * 21 - u * 9)
    cloud += .15 * np.sin(u * 57 + v * 32) + .09 * np.cos(u * 121 - v * 93)
    maps = {}

    # Register each PNG with the correct Blender color interpretation. Data maps
    # bypass sRGB, while foliage keeps an independent, straight alpha channel.
    def save(name, rgb, alpha=None, linear=False):
        if np.ndim(rgb) == 1:
            rgb = np.broadcast_to(rgb, (n, n, 3))
        if alpha is None:
            alpha = np.ones(rgb.shape[:2], dtype=np.float32)
        path = directory / (name + ".png")
        png(path, np.concatenate((rgb, alpha[:, :, None]), axis=2))
        image = bpy.data.images.load(str(path), check_existing=False)
        image.name = name
        image.colorspace_settings.name = "Non-Color" if linear else "sRGB"
        image.alpha_mode = "STRAIGHT" if name == "leaf" else "CHANNEL_PACKED"
        maps[name] = image

    # Add one scalar weathering field to an artist-chosen RGB palette.
    def colored(base, variation):
        return np.stack([np.clip(c + variation, 0, 1) for c in base], axis=2)

    # Convert height gradients to unit tangent-space normals, then encode [-1,1]
    # as [0,1]. Image rows grow downward, accounting for the Y-gradient sign.
    def normal(name, height, strength):
        dy, dx = np.gradient(height)
        normal = np.stack((-dx * strength, dy * strength, np.ones_like(dx)), axis=2)
        normal /= np.linalg.norm(normal, axis=2, keepdims=True)
        save(name, normal * .5 + .5, linear=True)

    # Distinct spatial frequencies create plaster pits, directional oak grain,
    # roof wear, stone variation, and staggered brick mortar rather than flat color.
    plaster = cloud * .13 + fine * .075
    pits = (fine < -.47) * -.085
    cracks = np.exp(-np.abs(np.sin(u * 20 + np.sin(v * 9) * 1.2)) * 130) * (np.sin(v * 6) > .4)
    save("lime_plaster", colored((.91, .86, .74), plaster + pits - cracks * .09))
    normal("plaster_normal", plaster + pits, 5)
    knots = np.sin(np.sqrt(((u - .3) * 8) ** 2 + ((v - .58) * 2) ** 2) * 47)
    knot_mask = np.exp(-((u - .3) / .16) ** 2 - ((v - .58) / .19) ** 2)
    wood = grain * .095 + fine * .035 + cloud * .05 + knots * knot_mask * .055
    save("aged_oak", colored((.57, .40, .245), wood))
    normal("oak_normal", grain * .06 + knots * knot_mask * .025 + fine * .025, 9)
    shingle = grain * .075 + cloud * .085 + fine * .06 - np.exp(-np.abs(np.sin(u * 31 + v)) * 95) * .065
    save("weathered_shingle", colored((.50, .48, .415), shingle))
    normal("shingle_normal", shingle, 6)
    stone = cloud * .17 + fine * .09 + .035 * np.sin(u * 91 + np.sin(v * 37) * 3)
    save("hewn_limestone", colored((.69, .66, .55), stone))
    normal("stone_normal", stone, 8)
    brickrow = np.floor(v * 18)
    mortar = (np.mod(v * 18, 1) < .075) | (np.mod(u * 8 + (brickrow % 2) * .5, 1) < .045)
    brick = colored((.63, .35, .245), cloud * .09 + fine * .055)
    brick[mortar] = (.59, .55, .46)
    save("old_brick", brick)
    normal("brick_normal", mortar.astype(np.float32) * -.3 + fine * .04, 4)
    # Fine canvas weave and sparse seam darkening complement the modeled cloth folds.
    weave = .014 * np.sin(u * math.tau * 240) * np.sin(v * math.tau * 210)
    seams = np.exp(-np.abs(np.sin(u * math.pi * 9)) * 180) * .045
    sail = colored((.94, .905, .79), fine * .015 + cloud * .023 + weave - seams)
    save("cream_canvas", sail)
    normal("canvas_normal", weave + fine * .008, 3)
    # Small foliage maps keep thousands of cards cheap. Alpha is never color-decoded.
    v, u = np.mgrid[0:256, 0:256].astype(np.float32) / 256
    x, y = (u - .5) * 2, (v - .5) * 2
    q = (x / np.maximum(.03, (1 - y * y) * .83)) ** 2 + y * y
    alpha = np.clip((1 - q) * 28 + .5, 0, 1)
    veins = np.exp(-np.abs(x) * 95) * .11 + (np.mod(y * 7 + np.abs(x) * 4, 1) < .06) * .035
    leaf = np.stack((.20 + veins + .05 * v, .42 + veins + .12 * (1 - v), .065 + veins * .4), axis=2)
    save("leaf", leaf, alpha)
    # Small-scale water perturbations are a baked normal map; large waves remain
    # actual closed-mesh geometry in model.py.
    v, u = np.mgrid[0:512, 0:512].astype(np.float32) / 512
    x = .17 * np.cos(math.tau * (5 * u + 2 * v)) + .05 * np.sin(math.tau * (13 * u - 3 * v))
    y = .13 * np.sin(math.tau * (6 * v - u)) + .04 * np.cos(math.tau * (7 * u + 11 * v))
    z = np.sqrt(1 - x * x - y * y)
    save("water_ripples", np.stack((x, y, z), axis=2) * .5 + .5, linear=True)
    return maps


# Use the verified fixture's supported Principled-node patterns. These authoring
# graphs export into ordinary glTF factors/maps, not procedural renderer shaders.
def make_materials(contract, images):
    mats = {}

    # Share baked images across material variations; multiply by a constant tint
    # and optionally attach a tangent-space normal map.
    def textured(name, image, tint, roughness=.75, normal=None, scale=.45, metal=0, two_sided=False):
        mat, pbr, _ = contract.material(name, tint, roughness, metal, two_sided)
        if image:
            tex = contract.image_node(mat, images[image])
            multiply = mat.node_tree.nodes.new("ShaderNodeMixRGB")
            multiply.blend_type = "MULTIPLY"
            multiply.inputs[0].default_value = 1
            multiply.inputs[2].default_value = (*tint, 1)
            mat.node_tree.links.new(tex.outputs["Color"], multiply.inputs[1])
            mat.node_tree.links.new(multiply.outputs[0], pbr.inputs["Base Color"])
        if normal:
            contract.normal_map(mat, pbr, images[normal], scale)
        mats[name] = mat
        return mat, pbr

    # Palette variants break up repetitive roofs, beams, and masonry without
    # allocating a separate large texture for every object.
    for i, tint in enumerate(((.92, .85, .69), (.86, .71, .48), (.49, .67, .59), (.91, .68, .48), (.61, .68, .65))):
        textured("plaster" + str(i), "lime_plaster", tint, .88, "plaster_normal", .36)
    for i, tint in enumerate(((.29, .21, .15), (.48, .38, .25), (.70, .56, .39), (.87, .78, .62), (.27, .40, .37))):
        textured("wood" + str(i), "aged_oak", tint, .66, "oak_normal", .38)
    for i in range(8):
        f = .54 + .055 * i
        textured("roof" + str(i), "weathered_shingle", (f * 1.04, f, f * .91), .83, "shingle_normal", .36)
    for i in range(7):
        f = .57 + i * .065
        textured("stone" + str(i), "hewn_limestone", (f, f * .97, f * .89), .90, "stone_normal", .46)
    textured("mortar", None, (.19, .22, .20), .95)
    textured("brick", "old_brick", (.85, .79, .66), .84, "brick_normal", .48)
    textured("canvas", "cream_canvas", (1, .85, .59), .88, "canvas_normal", .16, two_sided=True)
    textured("canvas_weathered", "cream_canvas", (.82, .87, .74), .91, "canvas_normal", .25, two_sided=True)
    textured("canvas_teal", "cream_canvas", (.19, .41, .32), .89, "canvas_normal", .3, two_sided=True)
    textured("canvas_seam", None, (.61, .49, .30), .88)
    textured("rope", "aged_oak", (.95, .81, .56), .89, "oak_normal", .35)
    textured("rigging", None, (.09, .075, .045), .82)
    textured("iron", None, (.065, .075, .073), .34, metal=.75)
    textured("brass", None, (.48, .32, .09), .29, metal=.8)
    textured("copper", "weathered_shingle", (.20, .47, .42), .48, metal=.40)
    textured("glass_dark", None, (.035, .11, .125), .13, metal=.26)
    textured("glass_light", None, (.13, .28, .30), .17, metal=.22)
    textured("recess", None, (.027, .035, .031), .85)
    textured("paint_red", "aged_oak", (.55, .12, .065), .65, "oak_normal", .25)
    textured("paint_blue", "aged_oak", (.16, .33, .43), .66, "oak_normal", .3)
    textured("white", None, (.86, .88, .81), .70, two_sided=True)
    textured("bird_dark", None, (.018, .025, .027), .77, two_sided=True)
    textured("flower_red", None, (.70, .12, .035), .8, two_sided=True)
    textured("flower_yellow", None, (.98, .61, .035), .8, two_sided=True)
    textured("flower_white", None, (.86, .84, .73), .8, two_sided=True)
    textured("seabed", "hewn_limestone", (.69, .89, .81), .96, "stone_normal", .5)
    textured("distant_hill", None, (.12, .21, .16), .93)
    # ROUND on alpha is an exporter-recognized MASK graph, not alpha blending.
    for i, color in enumerate(((.69, .85, .42), (.91, .92, .43), (.39, .67, .35), (.55, .77, .33))):
        mat, pbr = textured("leaf" + str(i), "leaf", color, .86, two_sided=True)
        tex = next(n for n in mat.node_tree.nodes if n.type == "TEX_IMAGE")
        tex.extension = "EXTEND"
        rounding = mat.node_tree.nodes.new("ShaderNodeMath")
        rounding.operation = "ROUND"
        mat.node_tree.links.new(tex.outputs["Alpha"], rounding.inputs[0])
        mat.node_tree.links.new(rounding.outputs[0], pbr.inputs["Alpha"])
    mat, pbr = textured("lantern_glow", None, (.55, .27, .05), .35)
    pbr.inputs["Emission Color"].default_value = (1, .53, .14, 1)
    pbr.inputs["Emission Strength"].default_value = 2.2
    # The water is the only dielectric volume. Its white surface tint leaves
    # depth coloration to Beer absorption over actual traveled world distances.
    mat, pbr = textured("water", None, (1, 1, 1), .085, "water_ripples", .40)
    pbr.inputs["Transmission Weight"].default_value = 1
    pbr.inputs["IOR"].default_value = 1.333
    pbr.inputs["Thin Wall"].default_value = False
    output = next(n for n in mat.node_tree.nodes if n.type == "OUTPUT_MATERIAL")
    absorption = mat.node_tree.nodes.new("ShaderNodeVolumeAbsorption")
    absorption.inputs["Color"].default_value = (.22, .76, .81, 1)
    absorption.inputs["Density"].default_value = 1 / 9
    mat.node_tree.links.new(absorption.outputs[0], output.inputs["Volume"])
    # Keep the exporter-recognized thickness metadata group. The unlinked
    # Occlusion socket prevents group pruning but does not export an AO texture.
    group = bpy.data.node_groups.new("glTF Material Output", "ShaderNodeTree")
    group.interface.new_socket(name="Occlusion", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 1
    group.interface.new_socket(name="Thickness", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 7
    metadata = mat.node_tree.nodes.new("ShaderNodeGroup")
    metadata.node_tree = group
    metadata.inputs["Thickness"].default_value = 7
    return mats


def make_sky(path):
    """A shaded field of original cumulus lobes, not a sampled environment."""
    # Latitude/longitude pixels use the renderer's Y-up convention. A smooth
    # blue gradient supplies clear sky; overlapping analytic lobes supply clouds.
    w, h = 2048, 1024
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    theta, phi = math.pi * (v + .5) / h, math.tau * ((u + .5) / w - .5)
    elevation = np.cos(theta)
    t = np.clip(elevation, 0, 1) ** .55
    rgb = np.stack([a * (1 - t) + b * t for a, b in zip((.24, .74, 1.1), (.06, .28, .62))], axis=2)
    rng = np.random.default_rng(4119)
    density = np.zeros_like(t)
    light = np.zeros_like(t)
    # The camera sees approximately phi=-pi/2. Give that hemisphere tall cloud banks.
    for center, altitude, spread in ((-2.12, .18, .30), (-1.55, .37, .34), (-.98, .20, .27),
                                     (-.42, .29, .38), (.60, .23, .40), (1.7, .35, .35), (2.65, .19, .44)):
        for k in range(17):
            az = center + rng.uniform(-spread, spread)
            el = altitude + rng.uniform(-.11, .13)
            sx, sy = rng.uniform(.045, .13), rng.uniform(.035, .095)
            dx = np.arctan2(np.sin(phi - az), np.cos(phi - az))
            dy = elevation - el
            shape = np.exp(-1.7 * ((dx / sx) ** 2 + (dy / sy) ** 2))
            density = np.maximum(density, shape)
            lit = np.clip(.48 + .55 * dy / sy - .34 * dx / sx, .06, 1.1)
            light = np.maximum(light, shape * lit)
    # Soften and perturb cloud boundaries, then shade the lobes with a directional
    # light impression. There is deliberately no sun disk to duplicate the glTF sun.
    turbulence = .035 * np.sin(phi * 137 + elevation * 77) * np.sin(elevation * 153 - phi * 39)
    opacity = np.clip((density - .15 + turbulence) * 3.8, 0, 1)
    lit = np.clip(light / np.maximum(density, .001), .08, 1)
    clouds = np.stack([a + b * lit for a, b in zip((.22, .38, .44), (1.46, 1.25, .95))], axis=2)
    rgb = rgb * (1 - opacity[:, :, None]) + clouds * opacity[:, :, None]
    ground = np.clip(-elevation * 12, 0, 1)[:, :, None]
    rgb = rgb * (1 - ground) + np.array((.075, .095, .085)) * ground
    # RGBE stores three mantissas with one shared exponent. Write legal per-channel
    # RLE scanlines, keeping radiance above one rather than tonemapping the sky.
    maxima = np.max(rgb, axis=2)
    mantissa, exponent = np.frexp(maxima)
    rgbe = np.empty((h, w, 4), dtype=np.uint8)
    rgbe[:, :, :3] = np.minimum(255, rgb * (mantissa * 256 / maxima)[:, :, None]).astype(np.uint8)
    rgbe[:, :, 3] = exponent + 128
    data = bytearray(f"#?RADIANCE\n# Original analytical cumulus sky; CC0-1.0; no sun disk\nFORMAT=32-bit_rle_rgbe\n\n-Y {h} +X {w}\n".encode("ascii"))
    for row in rgbe:
        data.extend((2, 2, w >> 8, w & 255))
        for channel in range(4):
            for start in range(0, w, 128):
                block = row[start:start + 128, channel].tobytes()
                data.append(len(block))
                data.extend(block)
    path.write_bytes(data)
    # Independently decode the result as data to validate dimensions and finite
    # radiance; loading an image here is not a Blender render.
    decoded = bpy.data.images.load(str(path), check_existing=False)
    assert tuple(decoded.size) == (w, h) and decoded.is_float
    values = np.empty(w * h * 4, dtype=np.float32)
    decoded.pixels.foreach_get(values)
    assert np.all(np.isfinite(values)) and np.min(values) >= 0
    bpy.data.images.remove(decoded)
    return {"width": w, "height": h, "encoding": "Radiance RGBE, linear RGB, -Y +X",
            "direction_convention": "u=atan2(z,x)/(2*pi)+.5; v=acos(y)/pi", "has_sun_disk": False,
            "rotation_degrees": 0, "intensity": 1, "radiance_range": [float(rgb.min()), float(rgb.max())]}
