# SPDX-License-Identifier: CC0-1.0
"""Build the original harbor GLB/HDR bundle. Never opens the reference image.

Blender is an asset preparation tool only. Preview/final images must be produced
by the independent CPU renderer. The verified fixture is never regenerated here.
"""

from collections import Counter
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import struct
import sys
import time

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
import numpy as np


# Locate the bundle relative to this source, independent of Blender's working directory.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIXTURES = ROOT / "scenes" / "fixtures"
sys.path.insert(0, str(HERE))
import model
import surfaces


# Content hashes pin shipped assets and protect the already-verified fixture.
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Export-time assertions produce a failing Blender process via --python-exit-code.
def require(condition, message):
    if not condition:
        raise RuntimeError(message)


# Keep the first-gate exports byte-for-byte intact while separating their original
# sky and authoring source from the detailed harbor workflow. Never re-export them here.
def preserve_fixture():
    path = FIXTURES / "export_contract.manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    original_sky = next(a for a in manifest["artifacts"] if a["path"].endswith("/sky.hdr"))
    target = FIXTURES / "sky.hdr"
    if not target.exists():
        source = ROOT / original_sky["path"]
        require(digest(source) == original_sky["sha256"], "Original fixture sky hash mismatch; refusing replacement")
        shutil.copyfile(source, target)
    require(digest(target) == original_sky["sha256"], "Preserved fixture sky hash mismatch")
    old_source = next(a for a in manifest["artifacts"] if a["path"].endswith("/author.py"))
    old_source.update(path="scenes/fixtures/author.py", bytes=(FIXTURES / "author.py").stat().st_size,
                      sha256=digest(FIXTURES / "author.py"))
    original_sky["path"] = "scenes/fixtures/sky.hdr"
    manifest["environment"]["path"] = original_sky["path"]
    manifest["run_command_powershell"] = f'& "{bpy.app.binary_path}" --background --factory-startup --python-exit-code 1 --python "{FIXTURES / "author.py"}"'
    manifest["renderer_import_parity"] = "VERIFIED_BY_MAIN"
    manifest["deferred_harbor"]["status"] = "IMPORT_PARITY_PASSED"
    manifest["known_contract_hazards"][0] = "Main verified actual CMake Release importer parity and a custom CPU render before detailed authoring; a regenerated fixture must be revalidated."
    manifest["fixture_preservation"] = {"binary_exports_regenerated": False, "sky_sha256_unchanged": True,
                                         "gate_evidence": "User reports 318852 parity checks, all six Release CTests, visually verified 480x270 CPU render at 64 spp."}
    for artifact in manifest["artifacts"]:
        require(digest(ROOT / artifact["path"]) == artifact["sha256"], "Fixture artifact changed: " + artifact["path"])
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {a["path"]: a["sha256"] for a in manifest["artifacts"]}


# Author the reference-composed view and physical glTF light parameters. These
# become scene data; the C++ renderer contains no harbor-specific camera/light code.
def camera_and_lights():
    scene = bpy.context.scene
    scene.name = "Original_Cream_Sail_Harbor"
    scene.unit_settings.system = "METRIC"
    scene.render.resolution_x, scene.render.resolution_y = 1920, 1080
    scene.render.resolution_percentage = 100
    data = bpy.data.cameras.new("Reference")
    data.type, data.lens, data.sensor_width = "PERSP", 30, 36
    data.clip_start, data.clip_end = .05, 1500
    camera = bpy.data.objects.new("Reference", data)
    scene.collection.objects.link(camera)
    camera.location = (0, -30, 5.2)
    target = Vector((0, 12, 6.1))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = camera
    # The verified exporter SPEC conversion uses 683 and, for points, 4*pi.
    # Calibrate Blender powers so exported glTF intensities are the intended values.
    sun = bpy.data.lights.new("Warm_Afternoon_Sun", "SUN")
    sun.energy, sun.color, sun.normalize = 3.4 / 683, (1, .91, .75), True
    obj = bpy.data.objects.new(sun.name, sun)
    scene.collection.objects.link(obj)
    obj.location = (10, -26, 40)
    obj.rotation_euler = Vector((-.25, .65, -1)).to_track_quat("-Z", "Y").to_euler()
    lamp = bpy.data.lights.new("Inn_Lantern_Point", "POINT")
    lamp.energy, lamp.color, lamp.normalize = .9 * 4 * np.pi / 683, (1, .58, .24), True
    lamp.shadow_soft_size = 0
    obj = bpy.data.objects.new(lamp.name, lamp)
    scene.collection.objects.link(obj)
    obj.location = (14.6, 5.3, 3.3)
    bpy.context.view_layer.update()
    return {"name": "Reference", "resolution": [1920, 1080], "blender_z_up_position": list(camera.location),
            "blender_z_up_target": list(target), "lens_mm": 30, "sensor_width_mm": 36,
            "clip": [.05, 1500], "glTF_up": "+Y", "sun_serialized_intensity": 3.4,
            "sun_color_linear": [1, .91, .75], "sun_ray_direction_blender": [-.25, .65, -1], "point_serialized_intensity": .9}


# Project landmark bounding boxes to normalized screen coordinates for composition
# auditing. This uses geometry/camera math, not a Blender-rendered preview image.
def projected_bounds(objects):
    groups = {"cream_gaff_cutter": "Cream_Gaff_Cutter", "hero_inn": "Hero_Harbor_Inn",
              "distant_tower": "Distant_Bell_Tower", "shipwright_house": "Shipwright_House",
              "foreground_rowboat": "Foreground_Rowboat"}
    result = {}
    for label, prefix in groups.items():
        points = []
        for obj in objects:
            if obj.name.startswith(prefix):
                points.extend(world_to_camera_view(bpy.context.scene, bpy.context.scene.camera, obj.matrix_world @ Vector(p)) for p in obj.bound_box)
        result[label] = {"normalized_screen_bounds_left_top_right_bottom": [min(p.x for p in points), 1 - max(p.y for p in points),
                                                                         max(p.x for p in points), 1 - min(p.y for p in points)]}
    return result


# Independently inspect the actual exported bytes, not just Blender objects.
# Validate primitive/frame data, images, materials, and the closed-water topology.
def validate_glb(path, allowed):
    raw = path.read_bytes()
    magic, version, total = struct.unpack_from("<III", raw)
    require(magic == 0x46546C67 and version == 2 and total == len(raw), "GLB header invalid")
    length, kind = struct.unpack_from("<II", raw, 12)
    require(kind == 0x4E4F534A, "Missing GLB JSON")
    doc = json.loads(raw[20:20 + length])
    offset = 20 + length
    length, kind = struct.unpack_from("<II", raw, offset)
    require(kind == 0x004E4942, "Missing GLB BIN")
    binary = memoryview(raw)[offset + 8:offset + 8 + length]
    require(set(doc.get("extensionsUsed", [])) <= allowed, "Unsupported extension exported")
    require(set(doc.get("extensionsRequired", [])) <= allowed, "Unsupported required extension")
    require(not doc.get("skins") and not doc.get("animations"), "Static scene contract violated")

    # Expose typed, strided array views into the GLB buffer without copying every attribute.
    def accessor(i):
        a = doc["accessors"][i]
        v = doc["bufferViews"][a["bufferView"]]
        dtype = {5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}[a["componentType"]]
        dim = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        size = np.dtype(dtype).itemsize
        return np.ndarray((a["count"], dim), dtype=dtype, buffer=binary,
                          offset=v.get("byteOffset", 0) + a.get("byteOffset", 0),
                          strides=(v.get("byteStride", size * dim), size))

    triangles, vertices, per_mesh, water_check = 0, 0, {}, None
    for mesh in doc["meshes"]:
        count = 0
        for primitive in mesh["primitives"]:
            require(primitive.get("mode", 4) == 4 and not primitive.get("targets"), "Non-triangle/static primitive")
            attrs = primitive["attributes"]
            require(set(attrs) == {"POSITION", "NORMAL", "TANGENT", "TEXCOORD_0"}, "Vertex attribute contract violated: " + mesh["name"])
            values = {k: accessor(i) for k, i in attrs.items()}
            require(all(np.isfinite(v).all() for v in values.values()), "Nonfinite vertex data")
            normals, tangents = values["NORMAL"], values["TANGENT"]
            require(np.max(np.abs(np.linalg.norm(normals, axis=1) - 1)) < .004, "Invalid normals: " + mesh["name"])
            require(np.max(np.abs(np.linalg.norm(tangents[:, :3], axis=1) - 1)) < .004, "Invalid tangents: " + mesh["name"])
            require(np.all(np.abs(np.abs(tangents[:, 3]) - 1) < .001), "Invalid tangent handedness")
            indices = accessor(primitive["indices"]).ravel()
            require(len(indices) % 3 == 0 and indices.max() < len(values["POSITION"]), "Invalid index buffer")
            faces = indices.reshape(-1, 3)
            positions = values["POSITION"]
            corners = positions[faces].astype(np.float64)
            cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
            require(np.all(np.einsum("ij,ij->i", cross, cross) > 1e-18), "Degenerate triangles: " + mesh["name"])
            count += len(faces)
            vertices += len(positions)
            # UV/normal seams duplicate vertices in glTF. Weld equal positions for
            # topological checks, then require paired opposite edges and positive volume.
            if mesh["name"].startswith("Water_Closed_Rippled_Harbor"):
                unique, inverse = np.unique(positions, axis=0, return_inverse=True)
                f = inverse[faces]
                edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
                undirected, group, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
                require(np.all(counts == 2), "Exported water is not watertight after seam welding")
                orientation = np.bincount(group, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
                require(np.all(orientation == 0), "Exported water edges have inconsistent orientation")
                volume = float(np.einsum("ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2])).sum() / 6)
                require(volume > 0, "Water orientation is inward")
                water_check = {"welded_vertices": len(unique), "edges": len(undirected), "triangles": len(faces),
                               "signed_volume_m3": volume, "closed_oriented_manifold": True}
        triangles += count
        per_mesh[mesh["name"]] = count
    require(150000 <= triangles <= 550000, f"Unexpected triangle budget: {triangles}")
    # Only water may introduce a volume; glazing/lanterns use opaque materials.
    volume_mats = []
    for material in doc["materials"]:
        require(material.get("alphaMode", "OPAQUE") in ("OPAQUE", "MASK"), "Alpha BLEND exported")
        require(not material.get("occlusionTexture"), "Occlusion map is outside the renderer contract")
        require(set(material.get("extensions", {})) <= allowed, "Unsupported material extension")
        if "KHR_materials_volume" in material.get("extensions", {}):
            volume_mats.append(material)
    require(len(volume_mats) == 1 and water_check is not None, "Exactly one closed water volume is required")
    ext = volume_mats[0]["extensions"]
    require(ext["KHR_materials_volume"]["thicknessFactor"] > 0 and "thicknessTexture" not in ext["KHR_materials_volume"], "Bad closed-water marker")
    require(abs(ext["KHR_materials_ior"]["ior"] - 1.333) < 1e-5, "Water IOR missing")
    require(ext["KHR_materials_transmission"]["transmissionFactor"] == 1, "Water transmission missing")
    # Every material image is embedded PNG data, keeping normal rendering self-contained.
    images = []
    for image in doc["images"]:
        require("uri" not in image and image["mimeType"] == "image/png", "GLB must embed every PNG")
        v = doc["bufferViews"][image["bufferView"]]
        data = binary[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]]
        require(data[:8] == b"\x89PNG\r\n\x1a\n", "Non-PNG image payload")
        w, h = struct.unpack_from(">II", data, 16)
        require(max(w, h) <= 2048, "Material map resolution exceeded")
        images.append({"name": image.get("name"), "width": w, "height": h, "sha256": hashlib.sha256(data).hexdigest()})
    require(len(doc["cameras"]) == 1 and doc["cameras"][0]["name"] == "Reference", "Reference camera missing")
    require(doc["cameras"][0]["type"] == "perspective", "Camera must be perspective")
    lights = doc["extensions"]["KHR_lights_punctual"]["lights"]
    return {"counts": {"triangles": triangles, "vertices_after_uv_normal_splits": vertices, "meshes": len(doc["meshes"]),
                        "nodes": len(doc["nodes"]), "materials": len(doc["materials"]), "images": len(images),
                        "textures": len(doc["textures"]), "cameras": len(doc["cameras"]), "punctual_lights": len(lights)},
            "generator": doc["asset"]["generator"], "extensions_used": sorted(doc.get("extensionsUsed", [])),
            "extensions_required": sorted(doc.get("extensionsRequired", [])), "camera": doc["cameras"][0],
            "lights": lights, "water_topology": water_check, "water_material": volume_mats[0],
            "embedded_images": images, "triangles_by_mesh": per_mesh,
            "validated": ["finite vertices and UVs", "unit normals and signed tangents", "indexed nondegenerate triangles",
                          "extension allowlist", "embedded PNGs <=2K", "no AO/COLOR_0/BLEND/animation/skin/morph",
                          "exported water welded manifold, paired opposite edges and positive signed volume"]}


def use_base_level_samplers(path):
    """Use explicit glTF core filters instead of requesting absent mip levels."""
    # Rewrite only JSON sampler declarations and chunk lengths. Preserve the BIN
    # tail containing geometry/images, and restore GLB's required 4-byte alignment.
    raw = path.read_bytes()
    length = struct.unpack_from("<I", raw, 12)[0]
    doc = json.loads(raw[20:20 + length])
    for sampler in doc.get("samplers", []):
        if sampler.get("minFilter") in (9984, 9986):
            sampler["minFilter"] = 9728
        elif sampler.get("minFilter") in (9985, 9987):
            sampler["minFilter"] = 9729
    payload = json.dumps(doc, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    payload += b" " * (-len(payload) % 4)
    tail = raw[20 + length:]
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, 20 + len(payload) + len(tail))
                     + struct.pack("<II", len(payload), 0x4E4F534A) + payload + tail)


# Preparation pipeline only: preserve the fixture, author maps/meshes/camera/sky,
# export once, validate bytes, and record provenance. Final pixels come from cpu_renderer.
def main():
    started = time.perf_counter()
    require(HERE.is_dir() and FIXTURES.is_dir(), "Missing authoring parent directories")
    require(bpy.app.version == (5, 2, 2), "Reverify export options before using a different Blender version")
    fixture_pins = preserve_fixture()
    # Reuse the installed-version-verified node/export helpers without invoking
    # the fixture module's __main__ entry point or regenerating its assets.
    spec = importlib.util.spec_from_file_location("verified_export_contract", FIXTURES / "author.py")
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.preferences.addon_enable(module="io_scene_gltf2")
    options, evidence, exporter_path = contract.verified_export_options()
    print("HARBOR: generating original material PNGs", flush=True)
    maps = surfaces.make_maps(HERE / "textures")
    mats = surfaces.make_materials(contract, maps)
    print("HARBOR: assembling geometry", flush=True)
    objects, source_triangles, features, source_water = model.build(mats)
    camera = camera_and_lights()
    bounds = projected_bounds(objects)
    print("HARBOR: source triangles " + str(sum(source_triangles.values())), flush=True)
    print("HARBOR: camera bounds " + json.dumps(bounds), flush=True)
    print("HARBOR: generating analytical cumulus HDR", flush=True)
    sky = surfaces.make_sky(HERE / "sky.hdr")
    path = HERE / "harbor.glb"
    options.update(filepath=str(path), export_format="GLB", export_copyright="Original harbor geometry, materials and analytical sky: CC0-1.0; " + contract.CC0)
    # The installed exporter raises KeyError for nondefault export_loglevel.
    # Keep its verified default, report warnings/errors without 1700 progress lines.
    export_log = io.StringIO()
    with redirect_stdout(export_log):
        outcome = bpy.ops.export_scene.gltf(**options)
    for line in export_log.getvalue().splitlines():
        if "WARNING" in line or "ERROR" in line:
            print(line, flush=True)
    require(outcome == {"FINISHED"}, "Blender GLB export did not finish")
    use_base_level_samplers(path)
    print("HARBOR: validating exported contract and water topology", flush=True)
    audit = validate_glb(path, contract.ALLOWED)
    require(audit["counts"]["triangles"] == sum(source_triangles.values()), "Exporter dropped or added source triangles")
    for file, pin in fixture_pins.items():
        require(digest(ROOT / file) == pin, "A shipped fixture artifact changed during harbor generation")
    # Pin both authoring source and generated deliverables, alongside licenses,
    # assumptions, budgets, and observed exported counts for later verification.
    artifacts = [*sorted(HERE.glob("*.py")), FIXTURES / "author.py", path, HERE / "sky.hdr", *sorted((HERE / "textures").glob("*.png"))]
    manifest = {"schema_version": 1, "stage": "detailed_harbor_exported", "name": "The Cream Sail Harbor",
                "not_a_blender_render": True, "reference_policy": "Visual guidance only. No reference pixels are read, sampled, copied, hashed or used as a texture/background by these sources.",
                "blender": {"version": bpy.app.version_string, "build_hash": bpy.app.build_hash.decode(), "exporter_path": str(exporter_path), "generator": audit["generator"]},
                "authoring_command_powershell": f'& "{bpy.app.binary_path}" --background --factory-startup --python-exit-code 1 --python "{HERE / "author.py"}"',
                "export_arguments": options, "verified_export_options": evidence,
                "post_export_steps": ["Set core glTF sampler minFilter to base-level nearest/linear, retaining all binary geometry/images unchanged. No absent mipmap hierarchy is requested."],
                "camera_composition": camera, "projected_landmarks": bounds, "features": features,
                "export_audit": audit, "water_authoring": source_water, "environment": dict(sky, path="scenes/harbor/sky.hdr"),
                "source_triangle_count": sum(source_triangles.values()), "authoring_elapsed_seconds": time.perf_counter() - started,
                "seeds": {"geometry": 630117, "maps": 630117, "sky": 4119},
                "preview": {"renderer": "custom cpu_renderer, not Blender", "status": "PENDING_VISUAL_INSPECTION",
                            "recommended_arguments": "--camera Reference --environment-rotation 0 --environment-intensity 1 --width 480 --height 270 --samples 24 --bounces 8 --threads 8 --seed 42 --exposure 0"},
                "known_caveats": ["Reference-composed original scene, not a pixel-exact reconstruction.",
                                  "Windows are opaque reflective glazing, not additional dielectric volumes.",
                                  "Water is the only closed dielectric; opaque bed, quay and hull geometry may be underwater.",
                                  "Nearest/linear base-level texture sampling is sufficient; no texture mip payloads are exported.",
                                  "Fixture binaries were preserved, not regenerated; original fixture sky is scenes/fixtures/sky.hdr.",
                                  "Blender mesh buffer re-exports are not promised byte-stable; pin delivered artifact hashes.",
                                  "No sun disk in the environment; illumination uses one directional sun plus weak inn point light and emissive lanterns."],
                "licenses": [{"scope": "All authoring source, generated geometry, textures and HDR", "spdx": "CC0-1.0", "url": contract.CC0,
                              "origin": "Original analytical/procedural modeling in this workspace; no external assets",
                              "notice": "To the extent possible under law, the authors waive all copyright and related or neighboring rights under CC0 1.0 Universal. No warranty.", "external_asset_urls": []},
                             {"scope": "Installed Blender authoring tool, not redistributed", "spdx": "GPL-3.0-or-later", "url": "https://www.blender.org/about/license/"},
                             {"scope": "Installed Khronos glTF exporter API, source inspected and invoked, not redistributed", "spdx": "Apache-2.0", "url": "https://github.com/KhronosGroup/glTF-Blender-IO"}],
                "fixture_preserved_pins": fixture_pins,
                "artifacts": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)} for p in artifacts]}
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("HARBOR_OK " + json.dumps({"counts": audit["counts"], "features": features, "sha256": digest(path),
                                     "seconds": manifest["authoring_elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
