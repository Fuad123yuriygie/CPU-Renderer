# SPDX-License-Identifier: CC0-1.0
"""Author the FIRST-GATE asset contract, not the detailed harbor or a render.

Run with Blender 5.2.2 --background --factory-startup --python-exit-code 1
--python scenes/fixtures/author.py. No reference image is opened by this program.
All geometry, PNG maps and the analytical HDR are original, dedicated to CC0.
"""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
from urllib.parse import unquote
import zlib

import bpy
from mathutils import Matrix, Quaternion, Vector


# All fixture outputs are workspace-local; this allowlist mirrors the renderer's
# supported material/lighting contract rather than every feature Blender can export.
ROOT = Path(__file__).resolve().parents[2]
HARBOR = ROOT / "scenes" / "harbor"
FIXTURES = ROOT / "scenes" / "fixtures"
MAPS = FIXTURES / "textures"
ALLOWED = {
    "KHR_texture_transform",
    "KHR_lights_punctual",
    "KHR_materials_emissive_strength",
    "KHR_materials_transmission",
    "KHR_materials_ior",
    "KHR_materials_volume",
}
CC0 = "https://creativecommons.org/publicdomain/zero/1.0/"


# Fail the preparation process clearly instead of shipping a partially valid fixture.
def require(condition, message):
    if not condition:
        raise RuntimeError(message)


# Byte digests pin source resources and allow embedded/external image comparison.
def sha256(data):
    return hashlib.sha256(data).hexdigest()


# Bound LDR channels and coverage/interpolation weights, not HDR radiance values.
def clamp(x):
    return max(0.0, min(1.0, x))


def write_png(path, size, sample):
    """Deterministic RGBA8 PNG, top row first; sample returns encoded channels."""
    # Write filter-zero scanlines and explicit PNG chunks so generated channel
    # values do not depend on Blender's image color-management/export settings.
    require(path.parent.is_dir(), f"Missing PNG parent: {path.parent}")
    raw = bytearray()
    for y in range(size):
        raw.append(0)
        for x in range(size):
            raw.extend(round(255 * clamp(c)) for c in sample((x + .5) / size, (y + .5) / size))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">2I5B", size, size, 8, 6, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


# Each small map exposes one contract detail: UV orientation, packed PBR channels,
# tangent-space normals, textured emission, or independent foliage alpha coverage.
def make_maps():
    def grid(u, v):
        # Asymmetric quadrant colors and a white top-left L reveal UV/image flips.
        colors = ((.78, .18, .12), (.18, .65, .26), (.12, .28, .82), (.88, .64, .13))
        c = colors[int(u >= .5) + 2 * int(v >= .5)]
        line = min((u * 8) % 1, (v * 8) % 1) < .055
        if (.07 < u < .10 and .08 < v < .38) or (.07 < u < .30 and .35 < v < .38):
            c = (.96, .96, .96)
        elif line:
            c = tuple(x * .35 for x in c)
        return (*c, 1.0)

    # glTF metallic-roughness maps use G for roughness and B for metallic.
    def metal_roughness(u, v):
        return (1.0, .16 + .64 * v, .55 + .45 * u, 1.0)

    # Unit analytic ripple normals, encoded into RGB without an sRGB conversion.
    def normal(u, v):
        x = .24 * math.cos(math.tau * (4 * u + v))
        y = .16 * math.sin(math.tau * (2 * v - u))
        z = math.sqrt(1 - x * x - y * y)
        return (.5 + .5 * x, .5 + .5 * y, .5 + .5 * z, 1.0)

    # Asymmetric warm bars test both emissive texture sampling and strength scaling.
    def emissive(u, v):
        bars = .24 if int(u * 9) % 3 == 0 else 1.0
        return (bars, bars * (.42 + .42 * v), bars * .18, 1.0)

    # Union several rotated ellipses and a stem into an original leaf-card mask.
    def foliage(u, v):
        coverage = 0.0
        vein = 0.0
        for side in (-1, 1):
            for i in range(4):
                cx, cy = .5 + side * (.12 + .025 * (i % 2)), .19 + .18 * i
                angle = side * (.48 + .10 * i)
                dx, dy = u - cx, v - cy
                a = math.cos(angle) * dx + math.sin(angle) * dy
                b = -math.sin(angle) * dx + math.cos(angle) * dy
                q = (a / .18) ** 2 + (b / .075) ** 2
                coverage = max(coverage, clamp((1 - q) * 16 + .5))
                if q < 1 and abs(b) < .008:
                    vein = .12
        if .10 < v < .92:
            coverage = max(coverage, clamp((.012 - abs(u - .5)) * 256))
        c = (.13 + vein, .32 + .18 * (1 - v) + vein, .055 + .06 * u)
        return (*c, coverage)

    specs = {
        "uv_grid": (256, grid, "sRGB", "RGBA: red/green top, blue/gold bottom; white L at top left"),
        "metal_roughness": (256, metal_roughness, "Non-Color", "R unused=1; G roughness=.16+.64*v; B metallic=.55+.45*u"),
        "ripple_normal": (256, normal, "Non-Color", "OpenGL tangent-space +Y normal; normalized analytic wave gradients"),
        "emissive_bars": (128, emissive, "sRGB", "Original asymmetric warm bars, multiplied by emissiveStrength=6"),
        "foliage_mask": (256, foliage, "sRGB", "RGB sRGB, alpha LINEAR; original elliptical leaves and stem, cutoff .42"),
    }
    # Load the generated maps as data, preserving sRGB/data-map distinctions and
    # checking alpha independently before the exporter ever sees the materials.
    images, records = {}, []
    for name, (size, sample, space, meaning) in specs.items():
        path = MAPS / (name + ".png")
        write_png(path, size, sample)
        image = bpy.data.images.load(str(path), check_existing=False)
        image.name = name
        image.colorspace_settings.name = space
        image.alpha_mode = "STRAIGHT" if name == "foliage_mask" else "CHANNEL_PACKED"
        require(tuple(image.size) == (size, size), f"PNG decode failed: {name}")
        alpha = image.pixels[3::4]
        if name == "foliage_mask":
            require(min(alpha) == 0 and max(alpha) == 1, "Foliage source lost its alpha coverage")
        images[name] = image
        records.append({"path": path.relative_to(ROOT).as_posix(), "width": size, "height": size,
                        "color_space": space, "channels": meaning, "source_sha256": sha256(path.read_bytes()),
                        "decoded_alpha_range": [min(alpha), max(alpha)],
                        "coverage_at_cutoff_0_42": sum(a >= .42 for a in alpha) / len(alpha)})
    return images, records


def make_sky(path):
    """Original Y-up equirectangular sky, RGBE RLE scanlines, no sun disk."""
    # Spherical coordinates select a clear-sky gradient and analytic cloud lobes;
    # the below-horizon region transitions to a low ground-light level.
    width, height = 512, 256
    require(path.parent.is_dir(), f"Missing HDR parent: {path.parent}")
    data = bytearray(f"#?RADIANCE\n# Original analytical sky; CC0-1.0; no sun disk\nFORMAT=32-bit_rle_rgbe\n\n-Y {height} +X {width}\n".encode("ascii"))
    minimum, maximum = float("inf"), 0.0
    for y in range(height):
        theta = math.pi * (y + .5) / height
        elevation = math.cos(theta)
        row = bytearray()
        for x in range(width):
            phi = math.tau * ((x + .5) / width - .5)
            if elevation >= 0:
                t = elevation ** .55
                rgb = [(1 - t) * a + t * b for a, b in zip((.90, 1.03, 1.15), (.16, .36, .72))]
                cloud = 0.0
                for center, altitude, spread in ((-2.6, .32, .40), (-1.45, .48, .34), (-.1, .24, .43), (1.15, .40, .38), (2.7, .22, .32)):
                    dphi = math.atan2(math.sin(phi - center), math.cos(phi - center))
                    lobes = 0.0
                    for k in range(4):
                        dx = dphi - (k - 1.5) * spread * .37
                        dy = elevation - altitude - .055 * math.sin(k * 2.2 + center)
                        lobes += math.exp(-2.0 * (dx / spread) ** 2 - 2.0 * (dy / (.10 + .018 * k)) ** 2)
                    cloud = max(cloud, clamp((lobes - .42) * .65))
                rgb = [(1 - cloud) * c + cloud * b for c, b in zip(rgb, (2.7, 2.6, 2.4))]
            else:
                t = clamp(-elevation * 9)
                rgb = [(1 - t) * a + t * b for a, b in zip((.90, 1.03, 1.15), (.045, .06, .075))]
            minimum, maximum = min(minimum, *rgb), max(maximum, *rgb)
            # One exponent shared by RGB preserves values above display white.
            fraction, exponent = math.frexp(max(rgb))
            scale = fraction * 256 / max(rgb)
            row.extend((*[min(255, int(c * scale)) for c in rgb], exponent + 128))
        # Radiance RLE stores each channel separately after a scanline header.
        data.extend((2, 2, width >> 8, width & 255))
        for channel in range(4):
            values = row[channel::4]
            for start in range(0, width, 128):
                block = values[start:start + 128]
                data.append(len(block))
                data.extend(block)
    path.write_bytes(data)
    decoded = bpy.data.images.load(str(path), check_existing=False)
    require(tuple(decoded.size) == (width, height) and decoded.is_float, "HDR must decode as floating point")
    require(all(math.isfinite(v) and v >= 0 for v in decoded.pixels[:]), "HDR contains invalid decoded pixels")
    bpy.data.images.remove(decoded)
    return {"path": path.relative_to(ROOT).as_posix(), "width": width, "height": height,
            "encoding": "Radiance RGBE; -Y +X; linear RGB", "has_sun_disk": False,
            "u": "atan2(direction.z,direction.x)/(2*pi)+0.5", "v": "acos(direction.y)/pi",
            "world_up": "+Y", "rotation_degrees": 0.0, "intensity": 1.0,
            "source_channel_range": [minimum, maximum], "decoded_by": "Blender image loader (not rendered)"}


# Minimal supported Principled material setup, also reused by the harbor author.
def material(name, color, roughness=.5, metallic=0.0, double_sided=False):
    mat = bpy.data.materials.new(name)
    mat.use_backface_culling = not double_sided
    nodes = mat.node_tree.nodes
    nodes.clear()
    pbr = nodes.new("ShaderNodeBsdfPrincipled")
    output = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(pbr.outputs["BSDF"], output.inputs["Surface"])
    pbr.inputs["Base Color"].default_value = (*color, 1.0)
    pbr.inputs["Roughness"].default_value = roughness
    pbr.inputs["Metallic"].default_value = metallic
    # Defaults matter: non-default specular/coat/sheen would add forbidden extensions.
    pbr.inputs["Specular IOR Level"].default_value = .5
    pbr.inputs["Specular Tint"].default_value = (1, 1, 1, 1)
    pbr.inputs["Coat Weight"].default_value = 0
    pbr.inputs["Sheen Weight"].default_value = 0
    pbr.inputs["Subsurface Weight"].default_value = 0
    pbr.inputs["Thin Film Thickness"].default_value = 0
    pbr.inputs["Anisotropic"].default_value = 0
    pbr.inputs["Alpha"].default_value = 1
    return mat, pbr, output


# Bind UVMap explicitly and optionally add the exporter's recognized texture transform.
def image_node(mat, image, wrap="REPEAT", filtering="Linear", mapping=None):
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    tex = nodes.new("ShaderNodeTexImage")
    tex.image, tex.extension, tex.interpolation = image, wrap, filtering
    uv = nodes.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap"
    source = uv.outputs["UV"]
    if mapping:
        transform = nodes.new("ShaderNodeMapping")
        transform.vector_type = "POINT"
        transform.inputs["Location"].default_value = mapping[0]
        transform.inputs["Rotation"].default_value = (0, 0, mapping[1])
        transform.inputs["Scale"].default_value = mapping[2]
        links.new(source, transform.inputs["Vector"])
        source = transform.outputs["Vector"]
    links.new(source, tex.inputs["Vector"])
    return tex


# Tangent-space normal-map nodes export to standard glTF normalTexture/scale data.
def normal_map(mat, pbr, image, strength):
    tex = image_node(mat, image)
    normal = mat.node_tree.nodes.new("ShaderNodeNormalMap")
    normal.space, normal.uv_map = "TANGENT", "UVMap"
    normal.inputs["Strength"].default_value = strength
    mat.node_tree.links.new(tex.outputs["Color"], normal.inputs["Color"])
    mat.node_tree.links.new(normal.outputs["Normal"], pbr.inputs["Normal"])


# Deliberately exercise every supported material feature while keeping unsupported
# shader lobes disabled, so this remains a small exporter/importer parity gate.
def make_materials(images):
    mats = {}
    mats["ground"], _, _ = material("Ground_Diffuse", (.20, .22, .24), .85)
    mat, pbr, _ = material("Diffuse_UVGrid", (1, 1, 1), .72)
    tex = image_node(mat, images["uv_grid"], "MIRROR", "Closest", ((.12, .18, 0), .31, (1.6, 1.2, 1)))
    mat.node_tree.links.new(tex.outputs["Color"], pbr.inputs["Base Color"])
    mats["diffuse"] = mat

    mat, pbr, _ = material("Metal_Roughness", (.66, .47, .22), 1, 1)
    tex = image_node(mat, images["metal_roughness"])
    separate = mat.node_tree.nodes.new("ShaderNodeSeparateColor")
    separate.mode = "RGB"
    mat.node_tree.links.new(tex.outputs["Color"], separate.inputs["Color"])
    mat.node_tree.links.new(separate.outputs["Green"], pbr.inputs["Roughness"])
    mat.node_tree.links.new(separate.outputs["Blue"], pbr.inputs["Metallic"])
    mats["metal"] = mat

    mat, pbr, _ = material("Normal_Mirrored", (.22, .38, .55), .27)
    normal_map(mat, pbr, images["ripple_normal"], .9)
    mats["normal"] = mat

    mat, pbr, _ = material("Emissive_Map", (.035, .035, .035), .6)
    tex = image_node(mat, images["emissive_bars"], "EXTEND")
    mat.node_tree.links.new(tex.outputs["Color"], pbr.inputs["Emission Color"])
    pbr.inputs["Emission Strength"].default_value = 6
    mats["emissive"] = mat

    # The installed exporter recognizes 1-(alpha<cutoff) as MASK. Legacy Blender
    # blend settings alone would not establish the desired glTF alpha semantics.
    mat, pbr, _ = material("Foliage_MASK", (1, 1, 1), .83, double_sided=True)
    tex = image_node(mat, images["foliage_mask"], "EXTEND", "Closest")
    mat.node_tree.links.new(tex.outputs["Color"], pbr.inputs["Base Color"])
    less = mat.node_tree.nodes.new("ShaderNodeMath")
    less.operation = "LESS_THAN"
    less.inputs[1].default_value = .42
    invert = mat.node_tree.nodes.new("ShaderNodeMath")
    invert.operation = "SUBTRACT"
    invert.inputs[0].default_value = 1
    mat.node_tree.links.new(tex.outputs["Alpha"], less.inputs[0])
    mat.node_tree.links.new(less.outputs[0], invert.inputs[1])
    mat.node_tree.links.new(invert.outputs[0], pbr.inputs["Alpha"])
    mats["foliage"] = mat

    # Separate surface transmission/IOR from non-scattering interior absorption.
    mat, pbr, output = material("Water_Volume", (1, 1, 1), .11)
    pbr.inputs["Transmission Weight"].default_value = 1
    pbr.inputs["IOR"].default_value = 1.333
    pbr.inputs["Thin Wall"].default_value = False
    normal_map(mat, pbr, images["ripple_normal"], .35)
    absorption = mat.node_tree.nodes.new("ShaderNodeVolumeAbsorption")
    absorption.inputs["Color"].default_value = (.20, .72, .81, 1)
    absorption.inputs["Density"].default_value = 1 / 3
    mat.node_tree.links.new(absorption.outputs["Volume"], output.inputs["Volume"])
    group = bpy.data.node_groups.new("glTF Material Output", "ShaderNodeTree")
    # An unlinked Occlusion input preserves this metadata group through 5.2 export.
    # It exports NO occlusionTexture. Thickness alone would be inlined away.
    group.interface.new_socket(name="Occlusion", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 1
    group.interface.new_socket(name="Thickness", in_out="INPUT", socket_type="NodeSocketFloat").default_value = .8
    metadata = mat.node_tree.nodes.new("ShaderNodeGroup")
    metadata.node_tree = group
    metadata.inputs["Thickness"].default_value = .8
    mats["water"] = mat
    return mats


# One UV-mapped plane for ground, cutout foliage, and textured emission probes.
def quad(name, vertices, mat, location=(0, 0, 0)):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], [(0, 1, 2, 3)])
    mesh.uv_layers.new(name="UVMap")
    for loop, uv in zip(mesh.uv_layers.active.data, ((0, 0), (1, 0), (1, 1), (0, 1))):
        loop.uv = uv
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    mesh.materials.append(mat)
    return obj


# Place distinguishable material probes, a parented mirrored transform, a closed
# water box, and a named camera/light setup. This is not the final harbor geometry.
def make_scene(mats):
    scene = bpy.context.scene
    scene.name = "Export_Contract_First_Gate"
    scene.render.resolution_x, scene.render.resolution_y = 1920, 1080
    scene.render.resolution_percentage = 100
    scene.unit_settings.system = "METRIC"
    quad("Ground", [(-7, -5, -.05), (7, -5, -.05), (7, 6, -.05), (-7, 6, -.05)], mats["ground"])
    for name, location, key in (("Diffuse_Probe", (-3.6, 1, 1), "diffuse"),
                                ("Metal_Probe", (-1.15, 1, 1), "metal"),
                                ("Mirrored_Nonuniform", (0, 0, 0), "normal")):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=1, location=location)
        obj = bpy.context.object
        obj.name = obj.data.name = name
        obj.data.uv_layers.active.name = "UVMap"
        obj.data.materials.append(mats[key])
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        if key == "normal":
            parent = bpy.data.objects.new("Transform_Parent", None)
            bpy.context.collection.objects.link(parent)
            parent.location = (1.4, 1.1, 1.28)
            parent.rotation_euler = (.09, -.13, .31)
            obj.parent = parent
            obj.location = (.15, -.1, 0)
            obj.rotation_euler = (.17, .35, -.22)
            obj.scale = (-.85, .65, 1.10)

    quad("Foliage_Card", [(-.9, 0, -1.25), (.9, 0, -1.25), (.9, 0, 1.25), (-.9, 0, 1.25)],
         mats["foliage"], (4.0, 1.3, 1.5))
    quad("Emissive_Panel", [(-.75, 0, -.4), (.75, 0, -.4), (.75, 0, .4), (-.75, 0, .4)],
         mats["emissive"], (1.5, 2.8, 2.9))
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-.6, -2.0, .45))
    water = bpy.context.object
    water.name = water.data.name = "Water_Closed_Box"
    for vertex in water.data.vertices:
        vertex.co.x *= 3.4
        vertex.co.y *= 2.1
        vertex.co.z *= .8
    water.data.uv_layers.active.name = "UVMap"
    water.data.materials.append(mats["water"])

    camera = bpy.data.cameras.new("Reference")
    camera.type, camera.lens, camera.sensor_width = "PERSP", 40, 36
    camera.clip_start, camera.clip_end = .05, 200
    obj = bpy.data.objects.new("Reference", camera)
    bpy.context.collection.objects.link(obj)
    obj.location = (0, -13, 7.4)
    obj.rotation_euler = (Vector((0, 0, 1)) - obj.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = obj

    # SPEC conversion is 683 lm/W, and 4*pi for point lights, verified installed source.
    for name, kind, strength, location, color in (
        ("Sun_Key", "SUN", 3 / 683, (0, 0, 6), (1.0, .91, .77)),
        ("Point_Fill", "POINT", 70 * 4 * math.pi / 683, (-3.0, -2.0, 4.5), (.72, .84, 1.0)),
    ):
        light = bpy.data.lights.new(name, kind)
        light.energy, light.color, light.normalize = strength, color, True
        obj = bpy.data.objects.new(name, light)
        bpy.context.collection.objects.link(obj)
        obj.location = location
        if kind == "SUN":
            obj.rotation_euler = Vector((.45, .30, -1)).to_track_quat("-Z", "Y").to_euler()
        else:
            light.shadow_soft_size = 0
    bpy.context.view_layer.update()


# Inspect the INSTALLED RNA/API options and enum values instead of assuming
# exporter versions share the same flags. Explicitly disable unsupported features.
def verified_export_options():
    import io_scene_gltf2

    options = {
        "check_existing": False, "export_image_format": "AUTO", "export_texture_dir": "textures",
        "export_keep_originals": False, "export_texcoords": True, "export_normals": True,
        "export_tangents": True, "export_materials": "EXPORT", "export_cameras": True,
        "export_lights": True, "export_import_convert_lighting_mode": "SPEC", "export_yup": True,
        "export_apply": True, "use_active_scene": True, "use_selection": False,
        "export_animations": False, "export_skins": False, "export_morph": False,
        "export_attributes": False, "export_vertex_color": "NONE", "export_all_vertex_colors": False,
        "export_active_vertex_color_when_no_material": False, "export_extras": False,
        "use_mesh_edges": False, "use_mesh_vertices": False, "export_gpu_instances": False,
        "export_draco_mesh_compression_enable": False, "export_meshopt_compression_enable": False,
        "export_use_gltfpack": False, "export_image_add_webp": False, "export_image_webp_fallback": False,
        "export_unused_images": False, "export_unused_textures": False,
        "export_hierarchy_flatten_objs": False, "export_hierarchy_full_collections": False,
        "export_pointer_animation": False, "will_save_settings": False,
        "export_copyright": "Original CPUrenderer fixture assets dedicated to CC0-1.0; " + CC0,
    }
    properties = bpy.ops.export_scene.gltf.get_rna_type().properties
    evidence = {}
    for name, value in options.items():
        require(name in properties, f"Installed exporter lacks option {name}")
        prop = properties[name]
        entry = {"type": prop.type, "default": prop.default, "description": prop.description}
        if prop.type == "ENUM":
            entry["choices"] = [item.identifier for item in prop.enum_items]
            require(value in entry["choices"], f"Unsupported value {name}={value}")
        evidence[name] = entry
    formats = [item[0] for item in io_scene_gltf2.get_format_items(None, bpy.context)]
    require({"GLB", "GLTF_SEPARATE"} <= set(formats), "Installed exporter lacks required formats")
    evidence["export_format"] = {"choices": formats, "verified_via": "installed get_format_items callback and two actual exports"}
    return options, evidence, Path(io_scene_gltf2.__file__).parent


# Decode both actual export containers and compare their semantic contents. These
# checks are independent of the custom C++ loader, whose parity is a separate gate.
def inspect_export(path):
    if path.suffix == ".glb":
        raw = path.read_bytes()
        magic, version, length = struct.unpack_from("<III", raw)
        require(magic == 0x46546C67 and version == 2 and length == len(raw), "Invalid GLB header")
        json_length, kind = struct.unpack_from("<II", raw, 12)
        require(kind == 0x4E4F534A, "Missing GLB JSON chunk")
        doc = json.loads(raw[20:20 + json_length])
        offset = 20 + json_length
        bin_length, kind = struct.unpack_from("<II", raw, offset)
        require(kind == 0x004E4942, "Missing GLB binary chunk")
        buffers = [raw[offset + 8:offset + 8 + bin_length]]
        require(all("uri" not in i and i["mimeType"] == "image/png" for i in doc["images"]), "GLB images must be embedded PNG")
    else:
        doc = json.loads(path.read_text(encoding="utf-8"))
        buffers = [(path.parent / unquote(b["uri"])).read_bytes() for b in doc["buffers"]]
        require(all("uri" in i and not i["uri"].startswith("data:") for i in doc["images"]), "glTF images must be external")
    require(doc["asset"]["version"] == "2.0", "Expected glTF 2.0")
    require(set(doc.get("extensionsUsed", [])) == ALLOWED, "All six and only six supported extensions must be exercised")
    require(set(doc.get("extensionsRequired", [])) <= ALLOWED, "Unsupported required extension")
    require(not doc.get("animations") and not doc.get("skins"), "Animation/skinning not allowed")

    # Read this fixture's packed/strided typed attributes directly from byte offsets.
    # Sparse/normalized cases are intentionally covered by the C++ runtime fixtures.
    def accessor(index):
        a = doc["accessors"][index]
        require(not a.get("sparse") and not a.get("normalized"), "Unexpected fixture accessor layout")
        view = doc["bufferViews"][a["bufferView"]]
        components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        fmt = "<" + {5121: "B", 5123: "H", 5125: "I", 5126: "f"}[a["componentType"]] * components
        stride = view.get("byteStride", struct.calcsize(fmt))
        offset = view.get("byteOffset", 0) + a.get("byteOffset", 0)
        values = [struct.unpack_from(fmt, buffers[view["buffer"]], offset + i * stride) for i in range(a["count"])]
        require(all(math.isfinite(v) for row in values for v in row), "Nonfinite accessor")
        return values

    meshes, triangle_count, semantic_geometry = [], 0, []
    water_closed = None
    for mesh in doc["meshes"]:
        primitives = []
        for p in mesh["primitives"]:
            require(p.get("mode", 4) == 4 and not p.get("targets"), "Static triangles only")
            require(set(p["attributes"]) == {"POSITION", "NORMAL", "TEXCOORD_0", "TANGENT"}, "Unexpected/missing vertex attributes")
            values = {k: accessor(v) for k, v in p["attributes"].items()}
            positions, normals, tangents = values["POSITION"], values["NORMAL"], values["TANGENT"]
            indices = [v[0] for v in accessor(p["indices"])]
            require(len(indices) % 3 == 0 and max(indices) < len(positions), "Invalid triangle indices")
            require(all(abs(Vector(n).length - 1) < .002 for n in normals), "Non-unit normals")
            require(all(abs(Vector(t[:3]).length - 1) < .002 and abs(abs(t[3]) - 1) < .001 for t in tangents), "Invalid tangent frames")
            for i in range(0, len(indices), 3):
                a, b, c = (Vector(positions[j]) for j in indices[i:i + 3])
                n = (b - a).cross(c - a)
                require(n.length > 1e-9, "Degenerate exported triangle")
                require(n.dot(sum((Vector(normals[j]) for j in indices[i:i + 3]), Vector())) > 0, "Normals disagree with winding")
            triangle_count += len(indices) // 3
            primitives.append({"vertices": len(positions), "triangles": len(indices) // 3,
                               "attributes": sorted(values), "material": doc["materials"][p["material"]]["name"],
                               "tangent_w_counts": dict(Counter(str(t[3]) for t in tangents)),
                               "local_bounds": [[min(v[k] for v in positions) for k in range(3)],
                                                [max(v[k] for v in positions) for k in range(3)]]})
            semantic_geometry.append({"name": mesh["name"], "material": p["material"], "values": values, "indices": indices})
            if mesh["name"] == "Water_Closed_Box":
                # Weld only equal exported positions; UV seams split glTF vertices.
                welded = {v: i for i, v in enumerate(sorted(set(positions)))}
                edges, directed, volume = Counter(), Counter(), 0.0
                for i in range(0, len(indices), 3):
                    ids = [welded[positions[j]] for j in indices[i:i + 3]]
                    a, b, c = (Vector(positions[j]) for j in indices[i:i + 3])
                    volume += a.dot(b.cross(c)) / 6
                    for a, b in zip(ids, ids[1:] + ids[:1]):
                        edges[tuple(sorted((a, b)))] += 1
                        directed[(a, b)] += 1
                require(all(n == 2 for n in edges.values()) and all(directed[(b, a)] == n for (a, b), n in directed.items()), "Water is not closed, consistently wound manifold")
                require(abs(volume - 3.4 * 2.1 * .8) < 1e-5, "Water must have outward positive volume")
                water_closed = {"welded_vertices": len(welded), "edges": len(edges), "signed_volume_m3": volume,
                                "closed_oriented_manifold": True, "dimensions_m": [3.4, 2.1, .8]}
        meshes.append({"name": mesh["name"], "primitives": primitives})

    # Assert the exporter retained the intended maps, alpha, volume parameters,
    # light calibration, and uniquely named perspective camera.
    materials = {m["name"]: m for m in doc["materials"]}
    for mat in materials.values():
        require(mat.get("alphaMode", "OPAQUE") in {"OPAQUE", "MASK"} and not mat.get("occlusionTexture"), "Unsupported alpha/AO")
        require(set(mat.get("extensions", {})) <= ALLOWED, "Unsupported material extension")
    foliage = materials["Foliage_MASK"]
    require(foliage.get("alphaMode") == "MASK" and foliage.get("doubleSided") is True, "Foliage must be two-sided MASK")
    require(abs(foliage.get("alphaCutoff", .5) - .42) < 1e-6, "MASK cutoff lost")
    water = materials["Water_Volume"]
    ext = water["extensions"]
    require(ext["KHR_materials_transmission"]["transmissionFactor"] == 1, "Transmission lost")
    require(abs(ext["KHR_materials_ior"]["ior"] - 1.333) < 1e-5, "Water IOR lost")
    volume = ext["KHR_materials_volume"]
    require(volume["thicknessFactor"] > 0 and not volume.get("thicknessTexture"), "Closed volume marker lost")
    require(abs(volume["attenuationDistance"] - 3) < 1e-5, "Attenuation distance lost")
    require(all(abs(a - b) < 1e-5 for a, b in zip(volume["attenuationColor"], (.2, .72, .81))), "Attenuation color lost")
    require(not water.get("doubleSided", False) and water_closed, "Water orientation contract lost")
    require("normalTexture" in materials["Normal_Mirrored"] and "metallicRoughnessTexture" in materials["Metal_Roughness"]["pbrMetallicRoughness"], "PBR maps lost")
    require("emissiveTexture" in materials["Emissive_Map"] and abs(materials["Emissive_Map"]["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"] - 6) < 1e-5, "Emission strength lost")
    require("KHR_texture_transform" in materials["Diffuse_UVGrid"]["pbrMetallicRoughness"]["baseColorTexture"]["extensions"], "UV transform lost")
    require(len(doc["cameras"]) == 1 and doc["cameras"][0]["name"] == "Reference" and doc["cameras"][0]["type"] == "perspective", "Named perspective camera lost")
    lights = doc["extensions"]["KHR_lights_punctual"]["lights"]
    require(sorted(l["type"] for l in lights) == ["directional", "point"], "Punctual lights lost")
    for light in lights:
        require(abs(light["intensity"] - (3 if light["type"] == "directional" else 70)) < 1e-4, "SPEC light conversion changed")

    nodes = doc["nodes"]
    world = {}

    # Independently compose glTF node matrices/TRS to derive a mirrored world-space
    # position/normal/tangent probe for the C++ importer's tests.
    def visit(index, parent):
        node = nodes[index]
        if "matrix" in node:
            local = Matrix([node["matrix"][i:i + 4] for i in range(0, 16, 4)]).transposed()
        else:
            q = node.get("rotation", [0, 0, 0, 1])
            local = Matrix.LocRotScale(Vector(node.get("translation", [0, 0, 0])), Quaternion((q[3], *q[:3])), Vector(node.get("scale", [1, 1, 1])))
        world[index] = parent @ local
        for child in node.get("children", []):
            visit(child, world[index])

    for root in doc["scenes"][doc.get("scene", 0)]["nodes"]:
        visit(root, Matrix.Identity(4))
    mirror_index = next(i for i, n in enumerate(nodes) if n.get("name") == "Mirrored_Nonuniform")
    mirror = world[mirror_index]
    require(mirror.to_3x3().determinant() < 0, "Mirrored transform was not retained")
    require(any(mirror_index in n.get("children", []) for n in nodes), "Transform parent was lost")
    p = doc["meshes"][nodes[mirror_index]["mesh"]]["primitives"][0]
    pos, norm, tangent = (accessor(p["attributes"][name])[0] for name in ("POSITION", "NORMAL", "TANGENT"))
    n = (mirror.to_3x3().inverted().transposed() @ Vector(norm)).normalized()
    t = mirror.to_3x3() @ Vector(tangent[:3])
    t = (t - n * n.dot(t)).normalized()
    mirror_probe = {"node": "Mirrored_Nonuniform", "determinant": mirror.to_3x3().determinant(),
                    "world_matrix_rows": [list(row) for row in mirror], "vertex_index": 0,
                    "world_position": list(mirror @ Vector(pos)), "world_normal": list(n),
                    "world_tangent": [*t, -tangent[3]], "reverse_triangle_winding": True}
    image_records = []
    for image in doc["images"]:
        if "uri" in image:
            data = (path.parent / unquote(image["uri"])).read_bytes()
        else:
            view = doc["bufferViews"][image["bufferView"]]
            start = view.get("byteOffset", 0)
            data = buffers[view["buffer"]][start:start + view["byteLength"]]
        require(data[:8] == b"\x89PNG\r\n\x1a\n", "Non-PNG texture")
        width, height = struct.unpack_from(">II", data, 16)
        require(max(width, height) <= 2048, "Texture budget exceeded")
        image_records.append({"name": image.get("name"), "width": width, "height": height, "sha256": sha256(data)})
    # Compare decoded meaning, not file bytes: equivalent GLB/external exports
    # naturally differ as containers, and mesh re-exports need not be byte-stable.
    semantic = {"geometry": semantic_geometry, "materials": doc["materials"], "nodes": nodes,
                "cameras": doc["cameras"], "lights": lights, "textures": doc["textures"],
                "images": image_records, "samplers": doc["samplers"]}
    counts = {"scenes": len(doc["scenes"]), "nodes": len(nodes), "meshes": len(meshes),
              "primitives": sum(len(m["primitives"]) for m in meshes), "triangles": triangle_count,
              "materials": len(materials), "images": len(doc["images"]), "textures": len(doc["textures"]),
              "samplers": len(doc["samplers"]), "cameras": len(doc["cameras"]), "punctual_lights": len(lights)}
    expected = {"scenes": 1, "nodes": 11, "meshes": 7, "primitives": 7, "triangles": 1602,
                "materials": 7, "images": 5, "textures": 6, "samplers": 4, "cameras": 1, "punctual_lights": 2}
    require(counts == expected, "Unexpected fixture counts")
    return {"path": path.relative_to(ROOT).as_posix(), "generator": doc["asset"]["generator"], "counts": counts, "meshes": meshes,
            "extensions_used": sorted(doc["extensionsUsed"]), "extensions_required": doc.get("extensionsRequired", []),
            "materials": doc["materials"], "camera": doc["cameras"][0], "lights": lights,
            "samplers": doc["samplers"], "images": image_records, "water_topology": water_closed,
            "mirror_probe": mirror_probe, "semantic_sha256": sha256(json.dumps(semantic, sort_keys=True).encode("utf-8")),
            "all_primitives_indexed": True, "uv_sets": [0], "sparse_accessors": False}


# First-gate workflow: reset Blender, generate original resources, export both
# forms, validate parity, and pin artifacts/options/licenses for reproducibility.
def main():
    require(ROOT.is_dir() and HARBOR.is_dir(), "Run the authoring source from the intended workspace")
    require(bpy.app.version == (5, 2, 2), "This contract was researched for installed Blender 5.2.2; reverify before using another exporter")
    for directory in (FIXTURES, MAPS):
        require(directory.parent.is_dir(), f"Missing output parent: {directory.parent}")
        directory.mkdir(exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.preferences.addon_enable(module="io_scene_gltf2")
    options, evidence, exporter = verified_export_options()
    images, maps = make_maps()
    sky = make_sky(FIXTURES / "sky.hdr")
    make_scene(make_materials(images))
    arguments, inspections = [], []
    # Every embedded/external image must preserve the exact authored PNG bytes.
    for suffix, export_format in ((".glb", "GLB"), (".gltf", "GLTF_SEPARATE")):
        path = FIXTURES / ("export_contract" + suffix)
        args = dict(options, filepath=str(path), export_format=export_format)
        require(bpy.ops.export_scene.gltf(**args) == {"FINISHED"}, f"Export failed: {path}")
        arguments.append(args)
        inspections.append(inspect_export(path))
        for image in inspections[-1]["images"]:
            original = next(m for m in maps if Path(m["path"]).stem == image["name"])
            require(original["source_sha256"] == image["sha256"], f"Exporter altered original PNG channels: {image['name']}")
    require(inspections[0]["semantic_sha256"] == inspections[1]["semantic_sha256"], "GLB/external glTF semantic parity failed")
    # Record the specific installed exporter source that informed this contract;
    # invoking/inspecting it does not redistribute its implementation.
    sources = ("__init__.py", "blender/exp/material/materials.py", "blender/exp/material/search_node_tree.py",
               "blender/exp/material/extensions/volume.py", "blender/exp/sampler.py", "blender/exp/lights.py",
               "blender/com/conversion.py")
    artifacts = [Path(__file__).resolve(), FIXTURES / "sky.hdr", FIXTURES / "export_contract.glb",
                 FIXTURES / "export_contract.gltf", FIXTURES / "export_contract.bin", *sorted(MAPS.glob("*.png"))]
    manifest = {
        "schema_version": 1, "stage": "first_gate_export_only", "renderer_import_parity": "PENDING_MAIN",
        "not_a_delivered_render": True, "reference_policy": "Visually inspected only. Authoring code never opens, samples, modifies, embeds or hashes Reference.png.",
        "blender": {"version": bpy.app.version_string, "version_tuple": list(bpy.app.version),
                    "build_hash": bpy.app.build_hash.decode(), "build_date": bpy.app.build_date.decode(),
                    "commit_date": bpy.app.build_commit_date.decode(), "gltf_generator": inspections[0]["generator"],
                    "binary": bpy.app.binary_path, "exporter_source_directory": str(exporter),
                    "exporter_source_sha256": {p: sha256((exporter / p).read_bytes()) for p in sources}},
        "run_command_powershell": f'& "{bpy.app.binary_path}" --background --factory-startup --python-exit-code 1 --python "{Path(__file__).resolve()}"',
        "export_arguments": arguments, "installed_option_evidence": evidence,
        "fixture": inspections, "maps": maps, "environment": sky,
        "self_checks": ["Actual operator properties and enum values verified before export", "Both actual exporter calls FINISHED",
                        "All six allowed extensions, no other extensions", "GLB embedded PNG and external glTF resources",
                        "Finite nondegenerate indexed triangles, UVs, unit normals/tangents", "Closed, outward water topology and positive signed volume",
                        "Mirrored hierarchical transform and expected world-space frame", "HDR decoded independently by Blender image loader",
                        "Source PNG bytes survive embedding/external export exactly, including independently decoded foliage alpha",
                        "GLB and external glTF decoded geometry/material/node/image semantic digests match"],
        "known_contract_hazards": [
            "Exporter-side checks are not custom-renderer parity; main must import and render both before harbor authoring resumes.",
            "Blender 5.2.2 uses AUTO for PNG, with PNG source files. There is no export_image_format=PNG enum.",
            "MASK is recognized from 1-(alpha<.42) nodes, not legacy blend_method. Alpha is linear even on an sRGB image.",
            "Volume needs an unlinked glTF Material Output Thickness=.8 plus Occlusion metadata socket to avoid 5.2 inline-node pruning. No AO map is exported.",
            "Volume attenuation color is linear (.2,.72,.81), attenuationDistance=3m, IOR=1.333, transmission=1. Thickness is a closed-body marker; integrate actual path lengths, not .8 at every hit.",
            "Water doubleSided=false does not authorize culling refractive exits. Keep geometric orientation; camera is outside all volumes; volumes do not overlap anything.",
            "Negative world determinant: reverse triangle winding, inverse-transpose normals, transform/orthogonalize tangents, multiply tangent.w by determinant sign.",
            "Blender exports glTF +Y up and flips UV V. Do not flip decoded top-left PNG rows or glTF UVs again. UV transform is already converted by the exporter.",
            "Sampler minification enums include mip filters, but only base-level images are supplied. Use their nearest/linear within-level behavior if no mip hierarchy exists.",
            "Exporter reports multiple image-node sampler warnings for foliage RGBA and packed metallic/roughness. Both paths use identical sampler settings and exact original PNG byte hashes are asserted; no conflicting sampler is authored.",
            "SPEC lights serialize to directional intensity 3 and point intensity 70; input Blender powers were calibrated by 683 and 4*pi. No sun disk exists in the HDR.",
            "Fixture exercises only indexed packed accessors and TEXCOORD_0, not sparse/strided/non-indexed accessors, JPEG, missing frames, or malformed resources. Main owns those tests.",
            "The sky is separate from glTF. Use rotation=0 and intensity=1, with the recorded Y-up equirectangular convention.",
        ],
        "licenses": [{"scope": "author.py and every generated fixture, map, mesh, material and HDR", "spdx": "CC0-1.0", "url": CC0,
                      "origin": "Original analytical/procedural code authored in this workspace; no downloaded assets or sampled images",
                      "dedication": "To the extent possible under law, the authors waive all copyright and related or neighboring rights in this original work under CC0 1.0 Universal. No warranty.",
                      "external_assets": []},
                     {"scope": "Blender authoring tool (installed, not redistributed)", "spdx": "GPL-3.0-or-later", "url": "https://www.blender.org/about/license/"},
                     {"scope": "Installed glTF exporter API used, source inspected but not copied or redistributed", "spdx": "Apache-2.0", "url": "https://github.com/KhronosGroup/glTF-Blender-IO"}],
        "deferred_harbor": {"status": "DO_NOT_START_BEFORE_MAIN_IMPORT_PARITY", "triangle_budget": [200000, 350000],
                            "maximum_material_map_dimension": 2048, "machine": "6 cores / 12 logical / 8 GB RAM",
                            "visual_composition": "Dominant cream gaff sailboat left; receding waterfront and distant tall tower; steep large half-timbered building right; curved stone quay foreground; turquoise closed rippled water; foliage, shingles, windows, chimneys, docks, small boats, birds and props. Organic silhouettes, not placeholder boxes."},
        "artifacts": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p.read_bytes())} for p in artifacts],
    }
    (FIXTURES / "export_contract.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("CONTRACT_OK " + json.dumps({"counts": inspections[0]["counts"], "semantic_sha256": inspections[0]["semantic_sha256"],
                                       "manifest": str(FIXTURES / "export_contract.manifest.json"), "renderer_import_parity": "PENDING_MAIN"}))


if __name__ == "__main__":
    main()
