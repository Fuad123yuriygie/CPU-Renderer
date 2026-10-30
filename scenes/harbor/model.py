# SPDX-License-Identifier: CC0-1.0
"""Original harbor geometry. Coordinates here are Blender Z-up, in meters."""

from collections import Counter
import math
import random

import bpy
from mathutils import Vector
from mathutils.geometry import tessellate_polygon
import numpy as np


# Repeatable procedural choices and manifest feature counters. G is the current
# mesh batch collector, initialized by build(); no geometry is created on import.
RNG = random.Random(630117)
FEATURES = Counter()
G = None


# Primitive helpers accept a point transform, with identity for world-space input.
def identity(p):
    return p


# Return a local-to-world point function: uniform scale, Z rotation, then translation.
# Blender authoring is Z-up; the glTF exporter performs the Y-up conversion later.
def frame(origin, angle=0, scale=1):
    c, s = math.cos(angle), math.sin(angle)

    def p(v):
        return (origin[0] + scale * (c * v[0] - s * v[1]),
                origin[1] + scale * (s * v[0] + c * v[1]), origin[2] + scale * v[2])
    return p


# Batch by named part/material rather than making one Blender object per stone,
# leaf, or shingle. Face-loop UVs and smoothing flags stay aligned with each face.
class Geometry:
    def __init__(self, mats):
        self.mats, self.batches = mats, {}

    # Rebase local face indices into the batch's vertex list and append corner UVs.
    def add(self, part, mat, vertices, faces, uv=None, smooth=False):
        key = (part, mat)
        b = self.batches.setdefault(key, [[], [], [], []])
        start = len(b[0])
        b[0].extend(vertices)
        for i, face in enumerate(faces):
            b[1].append(tuple(start + j for j in face))
            b[3].append(smooth if isinstance(smooth, bool) else smooth[i])
            if uv is not None:
                b[2].extend(uv[i])
            elif len(face) == 4:
                b[2].extend(((0, 0), (1, 0), (1, 1), (0, 1)))
            else:
                b[2].extend(((0, 0), (1, 0), (.5, 1)))

    # Materialize batched arrays into Blender meshes once, recording the triangle
    # count expected after export triangulation. Release Python batch storage afterward.
    def emit(self):
        objects, counts = [], {}
        for (part, mat), (verts, faces, uv, smooth) in self.batches.items():
            name = part + "__" + mat
            mesh = bpy.data.meshes.new(name)
            mesh.from_pydata(verts, [], faces)
            layer = mesh.uv_layers.new(name="UVMap")
            layer.data.foreach_set("uv", np.asarray(uv, dtype=np.float32).ravel())
            mesh.polygons.foreach_set("use_smooth", smooth)
            mesh.materials.append(self.mats[mat])
            mesh.update()
            obj = bpy.data.objects.new(name, mesh)
            bpy.context.collection.objects.link(obj)
            objects.append(obj)
            counts[name] = sum(len(f) - 2 for f in faces)
        self.batches.clear()
        return objects, counts


# Closed box with explicit outward face winding; used as one component of details.
def box(part, center, size, mat, p=identity):
    x, y, z = (s * .5 for s in size)
    cx, cy, cz = center
    vertices = [p((cx + a * x, cy + b * y, cz + c * z))
                for a, b, c in ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                                (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))]
    faces = [(0, 3, 2, 1), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)]
    G.add(part, mat, vertices, faces)


# One four-corner surface, with optional explicit UVs and smoothing.
def quad(part, points, mat, uv=None, smooth=False):
    G.add(part, mat, points, [(0, 1, 2, 3)], [uv] if uv else None, smooth)


# Sweep polygonal rings along a path. Local perpendicular frames orient each ring,
# accumulated arc length drives UVs, and optional caps close the endpoints.
def tube(part, points, radius, mat, sides=6, taper=1, caps=True):
    points = [Vector(x) for x in points]
    if len(points) < 2 or (points[-1] - points[0]).length < 1e-7:
        return
    verts, faces, uv, lengths = [], [], [], [0.0]
    for i in range(1, len(points)):
        lengths.append(lengths[-1] + (points[i] - points[i - 1]).length)
    for i, pt in enumerate(points):
        direction = (points[min(i + 1, len(points) - 1)] - points[max(0, i - 1)]).normalized()
        ref = Vector((0, 0, 1)) if abs(direction.z) < .95 else Vector((0, 1, 0))
        a, b = direction.cross(ref).normalized(), None
        b = direction.cross(a)
        r = radius * (1 + (taper - 1) * i / (len(points) - 1))
        for j in range(sides):
            v = pt + r * (a * math.cos(math.tau * j / sides) + b * math.sin(math.tau * j / sides))
            verts.append(tuple(v))
    for i in range(len(points) - 1):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            faces.append((a, b, b + sides, a + sides))
            uv.append(((j / sides, lengths[i]), ((j + 1) / sides, lengths[i]),
                       ((j + 1) / sides, lengths[i + 1]), (j / sides, lengths[i + 1])))
    if caps:
        # Triangular caps keep the mesh builder's UV contract simple.
        for ring, reverse in ((0, True), ((len(points) - 1) * sides, False)):
            for j in range(1, sides - 1):
                f = (ring, ring + j, ring + j + 1)
                faces.append(tuple(reversed(f)) if reverse else f)
                uv.append(((.5, .5), (0, 0), (1, 0)))
    G.add(part, mat, verts, faces, uv, smooth=True)


# A four-sided tube is an oriented timber/metal beam between arbitrary endpoints.
def beam(part, a, b, width, mat="wood0"):
    tube(part, [a, b], width * .71, mat, sides=4)


# Sample a parabolic sag between fixed endpoints before sweeping the rope mesh.
def rope(part, a, b, radius=.022, sag=.12, mat="rope", steps=10):
    points = []
    for i in range(steps + 1):
        t = i / steps
        v = Vector(a).lerp(Vector(b), t)
        v.z -= 4 * sag * t * (1 - t)
        points.append(v)
    tube(part, points, radius, mat, sides=5)


# Latitude rings plus pole fans form a smooth low-cost rounded volume.
def ellipsoid(part, center, size, mat, segments=10, rings=6, p=identity):
    cx, cy, cz = center
    sx, sy, sz = size
    verts = [p((cx, cy, cz + sz))]
    for j in range(1, rings):
        a = math.pi * j / rings
        for i in range(segments):
            b = math.tau * i / segments
            verts.append(p((cx + sx * math.sin(a) * math.cos(b), cy + sy * math.sin(a) * math.sin(b), cz + sz * math.cos(a))))
    verts.append(p((cx, cy, cz - sz)))
    faces = []
    for i in range(segments):
        faces.append((0, 1 + i, 1 + (i + 1) % segments))
    for j in range(rings - 2):
        a, b = 1 + j * segments, 1 + (j + 1) * segments
        for i in range(segments):
            n = (i + 1) % segments
            faces.append((a + i, b + i, b + n, a + n))
    last = 1 + (rings - 2) * segments
    for i in range(segments):
        faces.append((len(verts) - 1, last + (i + 1) % segments, last + i))
    G.add(part, mat, verts, faces, smooth=True)


# Circular swept detail for fittings, handles, clock rims, and rope coils.
def ring(part, center, radius, thickness, mat="iron", p=identity, segments=20):
    points = [p((center[0] + radius * math.cos(math.tau * i / segments), center[1],
                 center[2] + radius * math.sin(math.tau * i / segments))) for i in range(segments + 1)]
    # Deliberately avoid coincident path endpoints in tube().
    points[-1] = Vector(points[-1]).lerp(Vector(points[-2]), .0001)
    tube(part, points, thickness, mat, 5, caps=False)


# Construct beveled face/edge/corner polygons explicitly, then orient each outward.
# This adds readable edge highlights without expensive per-object modifiers.
def stone_block(part, center, size, mat, p=identity, bevel=.035):
    half = np.array(size) * .5
    corners = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    verts = []
    for c in corners:
        for axis in range(3):
            v = np.array(c) * (half - bevel)
            v[axis] = c[axis] * half[axis]
            verts.append(tuple(v))
    faces = []
    for axis in range(3):
        for sign in (-1, 1):
            ids = [i * 3 + axis for i, c in enumerate(corners) if c[axis] == sign]
            other = [i for i in range(3) if i != axis]
            ids.sort(key=lambda i: math.atan2(verts[i][other[1]], verts[i][other[0]]))
            faces.append(ids)
    for i, c in enumerate(corners):
        faces.append([i * 3, i * 3 + 1, i * 3 + 2])
        for axis in range(3):
            if c[axis] == 1:
                continue
            other = [a for a in range(3) if a != axis]
            d = list(c)
            d[axis] = 1
            j = corners.index(tuple(d))
            faces.append([i * 3 + other[0], i * 3 + other[1], j * 3 + other[1], j * 3 + other[0]])
    for face in faces:
        a, b, c = (Vector(verts[i]) for i in face[:3])
        if (b - a).cross(c - a).dot(sum((Vector(verts[i]) for i in face), Vector())) < 0:
            face.reverse()
    out = [p(tuple(center[i] + v[i] for i in range(3))) for v in verts]
    G.add(part, mat, out, faces)


# Layer a dark recess, reflective opaque glazing, frames, leading, sill, and
# optional shutters. These windows do not introduce additional refractive volumes.
def window(part, x, y, z, width, height, p, ornate=True, shutters=False):
    FEATURES["framed_windows"] += 1
    box(part, (x, y, z), (width + .20, .12, height + .20), "recess", p)
    box(part, (x, y - .085, z), (width - .12, .035, height - .12), RNG.choice(("glass_dark", "glass_light")), p)
    for xx in (x - width / 2, x + width / 2):
        box(part, (xx, y - .14, z), (.105, .18, height + .22), "wood1", p)
    for zz in (z - height / 2, z + height / 2):
        box(part, (x, y - .15, zz), (width + .22, .22, .115), "wood1", p)
    box(part, (x, y - .14, z), (.045, .11, height), "wood3", p)
    for zz in (z - height * .18, z + height * .19):
        box(part, (x, y - .14, zz), (width, .10, .045), "wood3", p)
    if ornate:
        # Diamond leading lies on the actual inset glazing, not a painted window.
        for side in (-1, 1):
            for row in range(4):
                xc, zc = x + side * width * .245, z - height * .37 + row * height * .245
                dx, dz = width * .21, height * .12
                for a, b in (((xc - dx, zc), (xc, zc + dz)), ((xc, zc + dz), (xc + dx, zc)),
                             ((xc + dx, zc), (xc, zc - dz)), ((xc, zc - dz), (xc - dx, zc))):
                    beam(part, p((a[0], y - .115, a[1])), p((b[0], y - .115, b[1])), .018, "iron")
    box(part, (x, y - .25, z - height * .5 - .09), (width + .38, .42, .12), "stone5", p)
    if shutters:
        for side in (-1, 1):
            xx = x + side * width * .77
            box(part, (xx, y - .05, z), (width * .40, .09, height), "paint_blue", p)
            for k in range(6):
                box(part, (xx, y - .13, z - height * .42 + k * height * .165), (width * .43, .10, .055), "wood1", p)


# Curved/sagged roof slopes carry staggered, overlapping individual shingles.
# More distant roofs use larger tiles to control geometry cost.
def roof(part, w, depth, eave, rise, p, detailed=True):
    # Map slope position and ridge distance to a slightly irregular roof surface.
    def at(side, t, y):
        x = side * (w * .5 + .44) * t
        z = eave + rise * (1 - t) + .52 * math.sin(t * math.pi) + .27 * t ** 9
        z += .045 * math.sin(y * 1.4 + w) - .10 * math.sin((y / depth + .5) * math.pi)
        return x, y, z

    length = math.hypot(w * .5 + .44, rise)
    rows = max(7, math.ceil(length / (.43 if detailed else .60)))
    cols = max(7, math.ceil((depth + .8) / (.43 if detailed else .61)))
    for side in (-1, 1):
        for row in range(rows):
            t0, t1 = row / rows, min(1.03, (row + 1.18) / rows)
            shift = .5 * (row % 2)
            for col in range(cols):
                y0 = -depth * .5 - .4 + (col - shift) * (depth + .8) / cols
                y1 = y0 + (depth + .8) / cols * .96
                t_end = min(1.04, t1 + RNG.uniform(-.012, .014))
                points = [at(side, t0, y0), at(side, t0, y1), at(side, t_end, y1), at(side, t_end, y0)]
                points = [(x, y, z + .03 + RNG.uniform(0, .012)) for x, y, z in points]
                # Correct side winding before baking world coordinates.
                if side == 1:
                    points.reverse()
                low = [(x, y, z - .055) for x, y, z in points]
                verts = [p(v) for v in points + low]
                faces = [(0, 1, 2, 3), (2, 6, 7, 3), (1, 5, 6, 2), (3, 7, 4, 0)]
                G.add(part + "_shingles", "roof" + str(RNG.randrange(8)), verts, faces)
                FEATURES["individual_shingles"] += 1
        for yy in (-depth * .5 - .49, depth * .5 + .44):
            pts = [p(at(side, i / 15, yy)) for i in range(16)]
            for a, b in zip(pts, pts[1:]):
                beam(part + "_bargeboards", a, b, .20, "wood1")
            tube(part + "_bargeboards", [Vector(v) + Vector((0, 0, .1)) for v in pts], .035, "wood3", 5)
        beam(part, p(at(side, 1, -depth * .5 - .4)), p(at(side, 1, depth * .5 + .4)), .19, "wood0")
    for k in range(max(8, int(depth / .35))):
        yy = -depth * .5 - .3 + (k + .5) * (depth + .6) / max(8, int(depth / .35))
        box(part + "_ridge", (0, yy, eave + rise + .1), (.32, .39, .18), "roof6", p)
    FEATURES["pitched_roofs"] += 1


# Brick shaft, projecting stone caps, and a recessed opening provide roof accents.
def chimney(part, x, y, bottom, height, p, width=.68):
    box(part, (x, y, bottom + height / 2), (width, width * .85, height), "brick", p)
    for z in (bottom + height - .3, bottom + height):
        stone_block(part, (x, y, z), (width + .24, width * .85 + .22, .16), "stone4", p)
    box(part, (x, y, bottom + height + .095), (width * .60, width * .48, .055), "recess", p)
    FEATURES["chimneys"] += 1


# Assemble a complete waterfront house from plaster, structural timber, openings,
# roof, and foundation. The hero flag adds closer-view architectural detail.
def house(name, center, w, depth, wall, rise, angle=0, variant=0, hero=False):
    p = frame(center, angle)
    front = -depth / 2
    # A pentagonal extrusion, with separately framed plaster bays and roof surfaces.
    verts = []
    for yy in (front, -front):
        verts.extend([p(v) for v in ((-w / 2, yy, 0), (w / 2, yy, 0), (w / 2 + .10, yy, wall),
                                    (.10, yy, wall + rise - .15), (-w / 2 - .09, yy, wall))])
    faces = [(0, 1, 2), (0, 2, 4), (2, 3, 4), (5, 9, 7), (5, 7, 6), (7, 9, 8),
             (0, 5, 6, 1), (1, 6, 7, 2), (2, 7, 8, 3), (3, 8, 9, 4), (4, 9, 5, 0)]
    # Project each facade onto its dominant plane for world-scaled plaster UVs.
    uv = []
    for f in faces:
        normal = (Vector(verts[f[1]]) - Vector(verts[f[0]])).cross(Vector(verts[f[2]]) - Vector(verts[f[0]]))
        axes = [k for k in range(3) if k != max(range(3), key=lambda k: abs(normal[k]))]
        uv.append([(verts[j][axes[0]] / 4, verts[j][axes[1]] / 4) for j in f])
    G.add(name + "_plaster", "plaster" + str(variant % 5), verts, faces, uv)
    # Floor/bay grids organize beams and openings, while small offsets and varied
    # palettes prevent the waterfront from becoming identical repeated boxes.
    floors = 3 if hero else (2 if wall > 4.1 else 1)
    levels = [wall * i / floors for i in range(floors + 1)]
    # Stone plinth and real staggered visible foundation blocks.
    for xx in np.arange(-w / 2, w / 2, .65):
        stone_block(name + "_plinth", (float(xx + .30), front - .045, .20), (.61, .40, .40), "stone" + str(RNG.randrange(7)), p)
    for yy in (front - .07, -front + .06):
        for zz in levels:
            box(name + "_timber", (0, yy, zz), (w + .27, .25, .21 if hero else .17), "wood0", p)
        for xx in np.linspace(-w / 2, w / 2, max(4, int(w / 1.55) + 1)):
            lean = RNG.uniform(-.07, .07)
            beam(name + "_timber", p((float(xx), yy, 0)), p((float(xx) + lean, yy, wall)), .20 if hero else .15)
        for side in (-1, 1):
            beam(name + "_gable", p((side * w / 2, yy, wall)), p((.1, yy, wall + rise - .14)), .24 if hero else .17, "wood1")
        beam(name + "_gable", p((.1, yy, wall)), p((.1, yy, wall + rise - .14)), .20, "wood0")
        for f in (.23, .46, .69):
            for side in (-1, 1):
                xx = side * w * .5 * f
                beam(name + "_gable", p((xx, yy, wall)), p((xx, yy, wall + rise * (1 - f) - .18)), .14, "wood1")
        beam(name + "_gable", p((-w * .23, yy, wall)), p((w * .23, yy, wall + rise * .45)), .13, "wood1")
        beam(name + "_gable", p((w * .23, yy, wall)), p((-w * .23, yy, wall + rise * .45)), .13, "wood1")
    # The camera sees the left flank as well as the principal gable.
    for side in (-1, 1):
        xx = side * (w / 2 + .065)
        for zz in levels:
            beam(name + "_timber", p((xx, front, zz)), p((xx, -front, zz)), .19, "wood0")
        for yy in np.linspace(front, -front, max(4, int(depth / 1.7))):
            beam(name + "_timber", p((xx, float(yy), 0)), p((xx, float(yy), wall)), .18, "wood0")
    bays = 4 if hero else (3 if w > 4.7 else 2)
    for floor in range(floors):
        z = wall * (floor + .5) / floors + .06
        for bay in range(bays):
            x = (bay + .5) * w / bays - w / 2
            if floor == 0 and bay == bays - 2:
                box(name + "_door", (x, front - .14, 1.15), (1.1, .12, 2.3), "recess", p)
                for k in range(8):
                    box(name + "_door", (x - .47 + k * .135, front - .24, 1.03), (.123, .11, 1.94), "wood" + str(1 + k % 3), p)
                for zz in (.40, 1.55):
                    box(name + "_door", (x, front - .32, zz), (1.05, .08, .065), "iron", p)
                ring(name + "_door", (x + .3, front - .35, 1.0), .085, .017, "brass", p)
                for k in range(3):
                    stone_block(name + "_steps", (x, front - .55 - k * .28, -.04 - k * .10), (1.45, .36, .20), "stone5", p)
            else:
                window(name + "_windows", x, front - .15, z, min(1.14, w / bays * .64), min(1.8, wall / floors * .62), p,
                       ornate=hero or center[1] < 22, shutters=(variant + bay + floor) % 3 == 0 and not hero)
            if floor > 0:
                left = (bay * w / bays - w / 2 + .15, front - .14, levels[floor] + .16)
                right = (left[0] + w / bays * .40, front - .14, levels[floor] + .65)
                beam(name + "_braces", p(left), p(right), .12, "wood1")
    window(name + "_attic", .10, front - .13, wall + rise * .35, min(1.2, w * .22), min(1.7, rise * .4), p, hero)
    # Side windows use a local facade rotated ninety degrees.
    sp = lambda v: p((-w * .5 + v[1] - .12, -v[0], v[2]))
    for floor in range(floors):
        for bay in range(3 if hero else 2):
            yy = (bay + .5) * depth / (3 if hero else 2) - depth / 2
            window(name + "_side_windows", yy, -.03, wall * (floor + .5) / floors, .95, min(1.65, wall / floors * .6), sp, hero)
    roof(name + "_roof", w, depth, wall, rise, p, detailed=center[1] < 25)
    chimney(name + "_chimney", -w * .24, depth * .23, wall + rise * .51, 2.2 if hero else 1.55, p, .83 if hero else .58)
    if hero:
        chimney(name + "_chimney", w * .22, -depth * .10, wall + rise * .57, 2.4, p, .71)
        # Broad balcony, projecting joists and carved knee braces.
        z, y = levels[2] - .14, front - .85
        box(name + "_balcony", (0, y, z), (w * .55, 1.25, .20), "wood2", p)
        for x in np.linspace(-w * .26, w * .26, 18):
            beam(name + "_balcony", p((float(x), y - .60, z)), p((float(x), y - .60, z + 1.05)), .055, "wood0")
        for zz in (z + .25, z + 1.04):
            beam(name + "_balcony", p((-w * .29, y - .62, zz)), p((w * .29, y - .62, zz)), .12, "wood1")
        for x in (-w * .23, w * .23):
            beam(name + "_balcony", p((x, front - .10, z - 1.1)), p((x, y - .45, z)), .17, "wood0")
        # Two inset gabled dormers on the visible roof slope.
        for yy in (-depth * .22, depth * .29):
            dp = lambda v, yy=yy: p((-w * .32 + v[1], yy - v[0], wall + rise * .34 + v[2]))
            box(name + "_dormers", (0, 0, .65), (1.65, 1.5, 1.3), "plaster2", dp)
            window(name + "_dormers", 0, -.80, .72, .95, 1.08, dp)
            roof(name + "_dormers", 1.75, 1.6, 1.3, 1.7, dp)
        # Hanging inn sign with geometry lettering-like carved marks, not a bitmap.
        beam(name + "_sign", p((w * .40, front, 3.1)), p((w * .40, front - 1.45, 3.1)), .085, "iron")
        box(name + "_sign", (w * .40, front - 1.13, 2.58), (.08, .75, .80), "wood1", p)
        for k in range(5):
            beam(name + "_sign", p((w * .40 - .055, front - 1.41 + k * .13, 2.35)), p((w * .40 - .055, front - 1.38 + k * .13, 2.73)), .035, "brass")
        porch = lambda v: p((v[0] + w * .10, v[1] + front - .92, v[2]))
        for xx in (-1.14, 1.14):
            beam(name + "_porch", porch((xx, -.68, .05)), porch((xx, -.68, 2.92)), .14, "wood1")
            beam(name + "_porch", porch((xx, -.68, 2.0)), porch((xx * .54, -.68, 2.88)), .11, "wood1")
        roof(name + "_porch", 2.65, 1.75, 2.94, 1.25, porch)
    FEATURES["individual_houses"] += 1


# Distant landmark: masonry shaft, cornices, belfry/louvers, clock, and copper spire.
def tower():
    name, p = "Distant_Bell_Tower", frame((-5.4, 41.5, 1.15), -.10)
    box(name, (0, 0, 6.2), (3.8, 3.8, 12.4), "stone5", p)
    for z, w in ((.3, 4.4), (8.8, 4.08), (12.3, 4.35), (12.6, 4.5), (15.7, 3.95)):
        stone_block(name + "_cornices", (0, 0, z), (w, w, .28), "stone6", p, .055)
    for side in range(4):
        a = math.pi * side / 2
        fp = lambda v, a=a: p((math.cos(a) * v[0] - math.sin(a) * v[1], math.sin(a) * v[0] + math.cos(a) * v[1], v[2]))
        for x in (-1.12, 1.12):
            box(name + "_pilasters", (x, -1.94, 6.4), (.24, .21, 11.6), "stone6", fp)
        for z in (3.4, 6.2, 10.1):
            for x in (-.58, .58):
                window(name + "_slits", x, -1.96, z, .47, 1.9, fp, ornate=False)
        for x in (-1.25, 0, 1.25):
            tube(name + "_belfry", [fp((x, -1.40, 12.75)), fp((x, -1.40, 15.7))], .13, "stone6", 8)
        box(name + "_belfry", (0, -1.29, 14.3), (2.4, .08, 2.4), "recess", fp)
        for z in np.arange(13.3, 15.5, .25):
            box(name + "_louvers", (0, -1.48, float(z)), (2.45, .19, .07), "copper", fp)
    # Clock face and dimensional hands/ticks on the forward wall.
    ring(name + "_clock", (0, -2.06, 10.1), .88, .075, "brass", p, 28)
    box(name + "_clock", (0, -2.01, 10.1), (1.38, .06, 1.38), "stone6", p)
    beam(name + "_clock", p((0, -2.12, 10.1)), p((.10, -2.12, 10.72)), .055, "iron")
    beam(name + "_clock", p((0, -2.13, 10.1)), p((-.44, -2.13, 9.92)), .065, "iron")
    for k in range(12):
        a = math.tau * k / 12
        beam(name + "_clock", p((.72 * math.cos(a), -2.13, 10.1 + .72 * math.sin(a))),
             p((.60 * math.cos(a), -2.13, 10.1 + .60 * math.sin(a))), .028, "iron")
    verts = [p(v) for v in ((-1.95, -1.95, 16), (1.95, -1.95, 16), (1.95, 1.95, 16), (-1.95, 1.95, 16), (0, 0, 20.1))]
    G.add(name + "_spire", "copper", verts, [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)])
    tube(name + "_finial", [p((0, 0, 19.8)), p((0, 0, 21.0))], .055, "brass", 8)
    ellipsoid(name + "_finial", (0, 0, 20.6), (.16, .16, .18), "brass", p=p)
    FEATURES["ornate_towers"] += 1


# Bilinearly interpolate four boundary corners, then add smooth billow and folds.
# The interior grid gives real curved normals/shadows; edge ropes remain geometry.
def sail(part, p, corners, material="canvas", billow=.72, nx=28, nz=34, seams=True):
    def point(s, t):
        bottom = Vector(corners[0]).lerp(Vector(corners[1]), s)
        top = Vector(corners[3]).lerp(Vector(corners[2]), s)
        v = bottom.lerp(top, t)
        v.x += billow * math.sin(math.pi * s) * math.sin(math.pi * t)
        v.x += .055 * math.sin(s * math.pi * 16 + t * 4) * math.sin(math.pi * t) * math.sin(math.pi * s)
        v.z -= .25 * math.sin(math.pi * s) * (1 - t)
        return tuple(v)
    verts = [p(point(i / nx, j / nz)) for j in range(nz + 1) for i in range(nx + 1)]
    faces, uv = [], []
    for j in range(nz):
        for i in range(nx):
            a = j * (nx + 1) + i
            faces.append((a, a + 1, a + nx + 2, a + nx + 1))
            uv.append(((i / nx, j / nz), ((i + 1) / nx, j / nz), ((i + 1) / nx, (j + 1) / nz), (i / nx, (j + 1) / nz)))
    G.add(part + "_billowed_cloth", material, verts, faces, uv, True)
    # Trace all four cloth boundaries for boltropes, then add raised seam lines.
    for fixed, horizontal in ((0, True), (1, True), (0, False), (1, False)):
        path = [p(point(i / 24, fixed) if horizontal else point(fixed, i / 24)) for i in range(25)]
        tube(part + "_boltropes", path, .029, "rope", 5)
    if seams:
        for s in np.linspace(.10, .90, 8):
            path = []
            for j in range(22):
                v = list(point(float(s), j / 21))
                v[0] += .013
                path.append(p(v))
            tube(part + "_seams", path, .012, "canvas_seam", 4)
    FEATURES["billowed_sails"] += 1


# One hull family supports detailed sailboats and smaller rowboats. Transforming
# local points lets differently scaled/oriented boats reuse construction logic.
def boat(name, origin, length=8.4, beamwidth=3.35, angle=-.90, scale=1, rig=True, canvas="canvas", hero=False):
    p = frame(origin, angle, scale)
    half = length / 2

    # Longitudinal u controls taper and sheer; vertical v curves keel into the rim.
    def hull(u, v, side, inset=0):
        y = (u * 2 - 1) * half
        breadth = beamwidth * .5 * max(.035, math.sin(math.pi * u) ** .68)
        breadth *= 1 - .10 * (1 - u)
        rim = 1.26 + .40 * abs(u * 2 - 1) ** 1.5
        x = side * (breadth * math.sin(v * math.pi * .5) ** .72 - inset)
        z = -.53 + (rim + .53) * v ** .78
        return (x, y, z)

    for side in (-1, 1):
        # Continuous dark hull under the narrow, separately modeled strake gaps.
        for row in range(11):
            vertices = [p(hull(i / 32, (row + t) / 11, side, .012)) for i in range(33) for t in (0, 1)]
            faces = [(i * 2, (i + 1) * 2, (i + 1) * 2 + 1, i * 2 + 1) for i in range(32)]
            if side == -1:
                faces = [tuple(reversed(f)) for f in faces]
            G.add(name + "_hull_backing", "wood0", vertices, faces, smooth=True)
            vertices = [p(hull(i / 32, (row + t) / 11, side)) for i in range(33) for t in (.025, .945)]
            G.add(name + "_carved_strakes", "wood" + str(1 + row % 3), vertices, faces, smooth=True)
            FEATURES["hull_strakes"] += 1
        for v, radius, mat in ((1.0, .095, "wood1"), (.83, .060, "wood0"), (.93, .036, "brass")):
            tube(name + "_gunwales", [p(hull(i / 40, v, side)) for i in range(41)], radius, mat, 7)
        for i in range(2, 21):
            u = i / 22
            a = hull(u, .89, side)
            b = hull(u, 1, side)
            b = (b[0], b[1], b[2] + .42)
            beam(name + "_rail_stanchions", p(a), p(b), .065, "wood1")
        rail = []
        for i in range(30):
            x, y, z = hull(.08 + .84 * i / 29, 1, side)
            rail.append(p((x, y, z + .42)))
        tube(name + "_handrails", rail, .055, "wood2", 6)
        for i in range(4, 20, 3):
            x, y, z = hull(i / 23, .77, side)
            # Dark inset ports set between heavy strakes, with raised surrounds.
            q = frame((x, y, z), math.pi * .5)
            portp = lambda v, q=q: p(q(v))
            box(name + "_ports", (0, 0, 0), (.24, .055, .14), "recess", portp)
    # Solid narrow stem, keel and dimensional deck boards.
    tube(name + "_stem", [p((0, -half - .05, -.45)), p((0, -half - .15, .45)), p((0, -half - .08, 1.65))], .11, "wood1", 7)
    beam(name + "_keel", p((0, -half + .4, -.52)), p((0, half - .3, -.52)), .15, "wood0")
    for i in range(28):
        u0, u1 = .035 + i * .93 / 28, .035 + (i + .92) * .93 / 28
        a, b = hull(u0, .965, -1), hull(u0, .965, 1)
        c, d = hull(u1, .965, 1), hull(u1, .965, -1)
        quad(name + "_deck_planks", [p(v) for v in (a, b, c, d)], "wood" + str(2 + i % 2))
    box(name + "_cabin", (0, half * .33, 1.54), (beamwidth * .55, 1.55, .62), "wood1", p)
    box(name + "_cabin", (0, half * .33, 1.92), (beamwidth * .63, 1.73, .13), "paint_red" if hero else "wood2", p)
    for x in (-beamwidth * .18, beamwidth * .18):
        box(name + "_cabin", (x, half * .33 - .79, 1.58), (.30, .04, .23), "glass_dark", p)
    for yy in (-half * .5, half * .7):
        for x in (-beamwidth * .32, beamwidth * .32):
            tube(name + "_cleats", [p((x, yy, 1.4)), p((x, yy, 1.67))], .055, "iron", 6)
            beam(name + "_cleats", p((x - .14, yy, 1.65)), p((x + .14, yy, 1.65)), .07, "iron")
    # Coiled deck lines and a pair of weathered spars/oars.
    for k in range(3):
        coilp = lambda v: p((v[0] - beamwidth * .27, v[2] + 1.8, 1.52 + v[1]))
        ring(name + "_rope_coils", (0, k * .028, 0), .21 - k * .035, .027, "rope", coilp, 24)
    # Rigged variants add spars, billowed cloth, and thin silhouette lines;
    # unrigged variants instead receive seats and oars.
    if rig:
        mast_y, mast_top = -length * .12, 14.1 if hero else length * 1.55
        tube(name + "_mast", [p((0, mast_y, 1.0)), p((.06, mast_y, mast_top))], .135, "wood2", 12, .40)
        y0, y1 = mast_y + .1, half * .86
        bottom, top = 2.45, mast_top * .73
        corners = [(0, y0, bottom), (.07, y1, bottom + .38), (.12, y1 - .3, top + 1.0), (0, y0, top)]
        sail(name + "_mainsail", p, corners, canvas, 1.15 if hero else .55, 32 if hero else 18, 40 if hero else 24, hero)
        tube(name + "_boom", [p((0, y0 - .28, bottom - .06)), p((.05, y1 + .3, bottom + .30))], .072, "wood2", 9)
        tube(name + "_gaff", [p((0, y0 - .10, top)), p((.12, y1, top + 1.08))], .065, "wood2", 9)
        # A narrow four-sided jib avoids a collapsed UV pole while retaining a pointed silhouette.
        jib = [(0, -half - .4, 1.95), (.02, mast_y - .22, 2.55), (0, mast_y - .2, top - .20), (0, mast_y - .34, top - .35)]
        sail(name + "_jib", p, jib, canvas, .47, 22 if hero else 14, 30 if hero else 20, False)
        tube(name + "_bowsprit", [p((0, mast_y - .2, 1.3)), p((0, -half - 1.4, 1.7))], .09, "wood1", 9, .4)
        # Standing rigging, halyards and shrouds form real silhouette lines.
        for target in ((0, -half - 1.35, 1.7), (-beamwidth * .45, -.4, 1.55), (beamwidth * .45, -.4, 1.55),
                       (-beamwidth * .40, half * .56, 1.6), (beamwidth * .40, half * .56, 1.6)):
            rope(name + "_standing_rig", p((.05, mast_y, mast_top - .25)), p(target), .017, .03, "rigging", 5)
        rope(name + "_halyard", p((0, mast_y, mast_top - .5)), p((.12, y1, top + 1.05)), .02, .05, "rope")
        for side in (-1, 1):
            for j in range(15 if hero else 8):
                t = .05 + j * .045
                toppt = Vector((.04, mast_y, mast_top * .9))
                a = Vector((side * beamwidth * .42, -.30, 1.5)).lerp(toppt, t)
                b = Vector((side * beamwidth * .42, 1.25, 1.5)).lerp(toppt, t)
                rope(name + "_ratlines", p(a), p(b), .013, .035, "rigging", 3)
            for u in (.37, .48, .59):
                x, y, z = hull(u, 1, side)
                ellipsoid(name + "_deadeyes", (x, y, z + .10), (.095, .055, .095), "wood0", 8, 4, p)
        # A small hand-authored pennant, with billowed vertices and no image source.
        vv = [p((.02 + .12 * math.sin(t * 4), mast_y + t * 1.3, mast_top - z * (.25 + t * .18))) for z in (0, 1) for t in (0, .33, .66, 1)]
        G.add(name + "_pennant", "paint_red" if hero else "white", vv, [(i, i + 4, i + 5, i + 1) for i in range(3)])
        FEATURES["rigged_boats"] += 1
    else:
        for y in (-length * .22, 0, length * .22):
            box(name + "_thwarts", (0, y, 1.3), (beamwidth * .81, .24, .095), "wood3", p)
        for side in (-1, 1):
            a, b = (side * .38, -.5, 1.45), (side * (beamwidth * .82), length * .50, 1.1)
            tube(name + "_oars", [p(a), p(b)], .035, "wood2", 6)
            box(name + "_oars", b, (.16, .6, .05), "wood3", p)
        FEATURES["rowboats"] += 1
    return p


# Hand-authored shoreline control points place the quay and receding waterfront.
COAST = [(28, -13), (19, -12), (13, -10), (9, -7.5), (6.8, -4), (6, 0), (5.2, 3.8),
         (2.8, 7.4), (-1, 10.8), (-5.5, 14), (-10, 18), (-14, 23.5), (-19, 32), (-24, 45)]


# Catmull-Rom interpolation turns the control polygon into a continuous quay curve.
def coast_points():
    pts = [Vector((*p, 0)) for p in COAST]
    out = []
    for i in range(len(pts) - 1):
        a, b, c, d = pts[max(0, i - 1)], pts[i], pts[i + 1], pts[min(len(pts) - 1, i + 2)]
        for j in range(8):
            t = j / 8
            v = .5 * ((2 * b) + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t * t + (-a + 3 * b - 3 * c + d) * t ** 3)
            out.append(v)
    out.append(pts[-1])
    return out


# Tessellate the land polygon, resample its shoreline by arc length, and place
# masonry/coping/paving/bollards in local frames aligned to that curve.
def quay():
    coast = coast_points()
    polygon = [Vector((p.x, p.y, 1.15)) for p in coast] + [Vector((-24, 70, 1.15)), Vector((50, 70, 1.15)), Vector((50, -13, 1.15))]
    # Land occupies the northeast side of the shoreline; the foreground is open water.
    triangles = tessellate_polygon([polygon])
    for indices in triangles:
        tri = tuple(polygon[i] for i in indices)
        a, b, c = tri
        if (b - a).cross(c - a).z < 0:
            tri = (a, c, b)
        G.add("Quay_land", "stone2", [tuple(v) for v in tri], [(0, 1, 2)], [[(v.x / 3, v.y / 3) for v in tri]])
    # Uniform spacing avoids clumped masonry where control points are close together.
    samples = []
    cumulative = 0.0
    next_at = 0.0
    for a, b in zip(coast, coast[1:]):
        delta = b - a
        length = delta.length
        while next_at <= cumulative + length:
            t = (next_at - cumulative) / length
            samples.append((a.lerp(b, t), delta.normalized()))
            next_at += .76
        cumulative += length
    for i, (pos, direction) in enumerate(samples):
        angle = math.atan2(direction.y, direction.x)
        p = frame((pos.x, pos.y, 0), angle)
        for row in range(4):
            shift = .37 if row % 2 else 0
            mat = "stone" + str(RNG.randrange(2, 7) if row > 1 else RNG.randrange(4))
            stone_block("Quay_handset_wall", (shift, 0, -.40 + row * .42), (.72 + RNG.uniform(-.035, .025), .65, .39), mat, p, .042)
            FEATURES["quay_wall_blocks"] += 1
        stone_block("Quay_coping", (.12, .03, 1.20 + RNG.uniform(-.018, .018)), (.76, .87, .23), "stone" + str(RNG.randrange(3, 7)), p, .047)
        # Irregular transverse cobbled courses only near the visible waterfront.
        if pos.y < 26:
            inward = Vector((direction.y, -direction.x, 0))
            for lane in range(1, 13 if pos.y < 8 else 7):
                q = pos + inward * (.43 + lane * .45)
                pf = frame((q.x, q.y, 1.17 + RNG.uniform(-.018, .020)), angle + RNG.uniform(-.045, .045))
                stone_block("Quay_cobblestones", (0, 0, 0), (.69 + RNG.uniform(-.05, .03), .405, .115),
                            "stone" + str(RNG.randrange(7)), pf, .022)
                FEATURES["cobblestones"] += 1
        if i % 6 == 0 and pos.y < 24:
            inward = Vector((direction.y, -direction.x, 0))
            q = pos + inward * .47
            tube("Quay_bollards", [(q.x, q.y, 1.30), (q.x, q.y, 1.98)], .075, "iron", 9, .82)
            ellipsoid("Quay_bollards", (q.x, q.y, 1.96), (.115, .115, .095), "brass", 10, 5)
            ring("Quay_mooring_rings", (q.x, q.y - .12, 1.6), .12, .025)
            FEATURES["bollards"] += 1
    return samples


# Large-scale geometric waves complement, rather than replace, the baked normal map.
def water_height(x, y):
    return .034 * math.sin(1.38 * x + 2.11 * y) + .019 * math.sin(3.10 * x - 1.72 * y) + .009 * math.cos(4.6 * y + x)


# A dense near-field grid and sparse distant border cover the visible water and
# off-camera reflections. The surface is closed with side walls and a bottom fan.
def water():
    xs = [-300, -200, -120, -80, -55] + list(np.linspace(-40, 40, 161)) + [55, 80, 120, 200, 300]
    ys = [-300, -150, -80, -42] + list(np.linspace(-20, 90, 221)) + [120, 170, 240, 400]
    nx, ny = len(xs), len(ys)
    verts = [(float(x), float(y), water_height(x, y)) for y in ys for x in xs]
    faces, uv, smooth = [], [], []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            f = (a, a + 1, a + nx + 1, a + nx)
            faces.append(f)
            uv.append([(verts[k][0] / 5, verts[k][1] / 5) for k in f])
            smooth.append(True)
    # Walk the perimeter consistently so side/bottom faces have outward winding.
    perimeter = list(range(nx)) + [j * nx + nx - 1 for j in range(1, ny)]
    perimeter += [(ny - 1) * nx + i for i in range(nx - 2, -1, -1)] + [j * nx for j in range(ny - 2, 0, -1)]
    bottom_start = len(verts)
    verts.extend((verts[k][0], verts[k][1], -7.0) for k in perimeter)
    center = len(verts)
    verts.append((0, 0, -7.0))
    for i, a in enumerate(perimeter):
        j = (i + 1) % len(perimeter)
        faces.append((a, bottom_start + i, bottom_start + j, perimeter[j]))
        uv.append([(0, 1), (0, 0), (1, 0), (1, 1)])
        smooth.append(False)
        faces.append((center, bottom_start + j, bottom_start + i))
        uv.append([(0, 0), (1, 0), (.5, 1)])
        smooth.append(False)
    # A closed oriented manifold uses every undirected edge twice, once each way.
    edges, directed = Counter(), Counter()
    for f in faces:
        for a, b in zip(f, f[1:] + f[:1]):
            edges[tuple(sorted((a, b)))] += 1
            directed[(a, b)] += 1
    assert all(n == 2 for n in edges.values()) and all(directed[(b, a)] == n for (a, b), n in directed.items())
    G.add("Water_Closed_Rippled_Harbor", "water", verts, faces, uv, smooth)
    # Real opaque submerged bottom; no second refractive body.
    bv, bf = [], []
    for j in range(31):
        for i in range(37):
            x, y = -90 + i * 5, -35 + j * 5
            bv.append((x, y, -3.65 + .24 * math.sin(x * .07) * math.cos(y * .11)))
    for j in range(30):
        for i in range(36):
            a = j * 37 + i
            bf.append((a, a + 1, a + 38, a + 37))
    G.add("Submerged_sandy_bed", "seabed", bv, bf, smooth=True)
    FEATURES["closed_water_volumes"] = 1
    return {"source_vertices": len(verts), "source_quads_and_triangles": len(faces), "edge_incidence_two": True,
            "oriented_manifold": True, "bottom_z": -7, "surface_height_range": [-.062, .062],
            "extent_blender_xy": [[-300, -300], [300, 400]], "attenuation_color": [.22, .76, .81],
            "attenuation_distance": 9, "ior": 1.333, "roughness": .085, "thickness_marker": 7,
            "opaque_underwater_objects": True, "nested_refractive_volumes": False}


# Planked decking, support piles, fasteners, rope, and a ladder provide dock-scale detail.
def dock(name, origin, length, angle):
    p = frame(origin, angle)
    for i in range(int(length / .23)):
        box(name + "_planks", (0, i * .23, 1.02 + RNG.uniform(-.015, .015)), (1.52, .215, .14), "wood" + str(1 + i % 3), p)
        FEATURES["dock_planks"] += 1
    for x in (-.55, .55):
        beam(name, p((x, -.1, .84)), p((x, length, .84)), .17, "wood0")
        for y in (.35, length * .48, length - .22):
            tube(name + "_piles", [p((x, y, -2.2)), p((x + .03, y, 1.74))], .14, "wood1", 9, .82)
            tube(name + "_pile_cap", [p((x, y, 1.62)), p((x, y, 1.75))], .15, "iron", 9)
    for y in np.arange(.4, length - .2, .8):
        for x in (-.57, .57):
            box(name + "_nails", (x, float(y), 1.102), (.032, .032, .012), "iron", p)
    rope(name + "_rope", p((-.55, .35, 1.62)), p((-.55, length - .22, 1.62)), .034, .3)
    for k in range(6):
        beam(name + "_ladder", p((.83, .6, .85 - k * .23)), p((.83, 1.25, .85 - k * .23)), .055, "wood2")
    for y in (.58, 1.27):
        beam(name + "_ladder", p((.83, y, -1)), p((.83, y, 1.22)), .075, "wood1")
    FEATURES["docks"] += 1


# Bulging wooden staves, metal hoops, and a plank lid are separate modeled surfaces.
def barrel(name, origin, scale=1):
    p = frame(origin, RNG.random() * math.tau, scale)
    for k in range(14):
        angles = [math.tau * (k + d) / 14 for d in (.035, .965)]
        vertices = []
        for z, r in ((0, .31), (.20, .36), (.53, .395), (.88, .36), (1.06, .31)):
            vertices.extend(p((r * math.cos(a), r * math.sin(a), z)) for a in angles)
        G.add(name + "_staves", "wood" + str(1 + k % 3), vertices, [(i * 2, i * 2 + 1, i * 2 + 3, i * 2 + 2) for i in range(4)])
    for z, r in ((.17, .36), (.54, .40), (.9, .36)):
        verts = [p((r * math.cos(math.tau * k / 20), r * math.sin(math.tau * k / 20), z + dz)) for dz in (-.036, .036) for k in range(20)]
        G.add(name + "_hoops", "iron", verts, [(k, (k + 1) % 20, (k + 1) % 20 + 20, k + 20) for k in range(20)])
    for x in np.arange(-.26, .29, .085):
        length = 2 * math.sqrt(max(.001, .30 ** 2 - x * x))
        box(name + "_lid", (float(x), 0, 1.06), (.078, length, .04), "wood2", p)
    FEATURES["barrels"] += 1


# Slatted seat/back with metal supports, positioned through a local quay frame.
def bench(origin, angle):
    p = frame(origin, angle)
    for y in (-.19, 0, .19):
        box("Quayside_benches", (0, y, .61), (1.85, .16, .09), "wood2", p)
    for z in (.93, 1.15):
        box("Quayside_benches", (0, .27, z), (1.85, .09, .15), "wood1", p)
    for x in (-.68, .68):
        for y in (-.18, .22):
            beam("Quayside_benches", p((x, y, .04)), p((x, y, .6)), .075, "iron")
        beam("Quayside_benches", p((x, .27, .05)), p((x, .27, 1.26)), .065, "iron")
    FEATURES["benches"] += 1


# Branches establish the silhouette; oriented alpha-MASK leaf cards populate
# several canopy lobes without using solid green proxy spheres.
def tree(name, origin, height=7, radius=2.3, leaves=700):
    x, y, z = origin
    tube(name + "_trunk", [(x, y, z), (x + .17, y - .08, z + height * .4), (x - .14, y + .18, z + height * .80)], height * .052, "wood1", 9, .17)
    lobes = []
    for k in range(9):
        a = math.tau * k / 9 + RNG.uniform(-.2, .2)
        d = radius * RNG.uniform(.40, .87)
        center = Vector((x + math.cos(a) * d, y + math.sin(a) * d, z + height * RNG.uniform(.65, .91)))
        lobes.append(center)
        tube(name + "_branches", [(x + .1, y, z + height * .4), tuple(center - Vector((0, 0, .4))), tuple(center)], .08, "wood1", 6, .25)
    for i in range(leaves):
        center = RNG.choice(lobes)
        a, c = RNG.uniform(0, math.tau), RNG.uniform(-1, 1)
        # Cube-root radius distributes samples through a volume rather than its center.
        r = radius * .53 * RNG.random() ** (1 / 3)
        pos = center + Vector((r * math.sqrt(1 - c * c) * math.cos(a), r * math.sqrt(1 - c * c) * math.sin(a), r * c * .8))
        normal = Vector((RNG.uniform(-1, 1), RNG.uniform(-1, 1), RNG.uniform(-.7, 1))).normalized()
        axis = normal.cross(Vector((0, 1, 0))).normalized()
        other = normal.cross(axis)
        size = RNG.uniform(.15, .29) * (height / 7) ** .35
        verts = [tuple(pos + axis * a * size + other * b * size * 1.65) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        quad(name + "_leaf_cards", verts, "leaf" + str(RNG.randrange(4)))
    FEATURES["trees"] += 1
    FEATURES["leaf_cards"] += leaves


# Small stems/petals add sparse foreground color and geometry-level scale cues.
def flowers(name, origin, spread=1, count=34):
    x, y, z = origin
    for i in range(count):
        xx, yy = x + RNG.uniform(-spread, spread), y + RNG.uniform(-spread * .36, spread * .36)
        height = RNG.uniform(.22, .68)
        tube(name + "_stems", [(xx, yy, z), (xx + .035, yy, z + height)], .009, "wood4", 4)
        mat = RNG.choice(("flower_red", "flower_red", "flower_yellow", "flower_white"))
        for k in range(5):
            a = math.tau * k / 5
            ellipsoid(name + "_petals", (xx + .045 * math.cos(a), yy + .045 * math.sin(a), z + height), (.042, .04, .035), mat, 5, 3)
        FEATURES["flowers"] += 1


# A rounded body and mirrored low-poly wing surfaces make readable distant silhouettes.
def bird(name, origin, scale, angle, white=False):
    p = frame(origin, angle, scale)
    ellipsoid(name, (0, 0, 0), (.13, .27, .10), "white" if white else "bird_dark", 8, 4, p)
    for side in (-1, 1):
        pts = [(0, 0, .03), (side * .43, -.09, .22), (side * .97, -.22, .15), (side * .49, .14, .14), (side * .19, .18, .02)]
        faces = [(0, 1, 4), (1, 3, 4), (1, 2, 3)]
        if side == -1:
            faces = [tuple(reversed(f)) for f in faces]
        # Opaque double-sided bird materials avoid duplicate coincident faces.
        G.add(name + "_wings", "white" if white else "bird_dark", [p(v) for v in pts], faces)
    FEATURES["birds"] += 1


# Arrange smaller harbor objects as original geometry; most individual pieces
# share batches/materials so this detail does not create thousands of scene nodes.
def props():
    dock("Near_wooden_jetty", (5.4, -1.2, 0), 4.1, 2.04)
    dock("Village_pier", (-2.2, 11.5, 0), 4.8, 2.75)
    dock("Old_boathouse_pier", (-10.2, 20.4, 0), 4.2, 2.7)
    for i, origin in enumerate(((9, 3.7, 1.2), (9.8, 3.95, 1.2), (14.2, 4.8, 1.2), (15.0, 4.9, 1.2),
                                 (4.3, 10.9, 1.2), (3.4, 10.8, 1.2), (-4.5, 17, 1.2), (-5.3, 16.8, 1.2),
                                 (17.6, -5.9, 1.2), (18.4, -5.9, 1.2), (10.9, -4.0, 1.2))):
        barrel("Harbor_barrel_" + str(i), origin, RNG.uniform(.70, 1.0))
    bench((12.9, -5.4, 1.20), -.2)
    bench((18.8, -8.4, 1.20), -.4)
    bench((6.9, 3.6, 1.20), -.9)
    for i, (x, y) in enumerate(((10.9, 5.1), (17.6, 6.1), (3.7, 11.7), (-5.4, 19.5))):
        box("Flower_planters", (x, y, 1.42), (1.5, .54, .45), "wood1")
        flowers("Flower_bed_" + str(i), (x, y, 1.66), .69, 26)
    flowers("Quay_wildflowers", (15.7, -5.7, 1.2), 1.45, 43)
    flowers("Inn_wall_flowers", (18.1, 5.1, 1.2), 1.05, 38)
    # Fish crates, open lobster cages, coiled ropes and cargo produce dock-level scale cues.
    for i, (x, y) in enumerate(((8.4, 2.8), (8.9, 3.4), (4.5, 9.5), (2.4, 11.2), (11.5, -4.5))):
        for side in (-1, 1):
            for k in range(4):
                box("Slatted_cargo_crates", (x, y + side * .30, 1.28 + k * .14), (.82, .06, .085), "wood2")
                box("Slatted_cargo_crates", (x + side * .41, y, 1.28 + k * .14), (.06, .60, .085), "wood1")
        for k in range(3):
            cp = lambda v, x=x, y=y: (x + v[0], y + v[2], 1.87 + v[1])
            ring("Cargo_rope_coils", (0, k * .027, 0), .24 - .035 * k, .026, "rope", cp, 22)
        FEATURES["cargo_crates"] += 1
    # Lanterns are opaque emissive amber inserts with iron frames, not extra glass volumes.
    for x, y, z in ((14.6, 5.5, 3.2), (11.5, 4.9, 3.0), (4.0, 11.6, 2.9), (-3.5, 17.1, 2.8)):
        box("Lanterns", (x, y, z), (.19, .19, .30), "lantern_glow")
        for xx in (-.115, .115):
            for yy in (-.115, .115):
                beam("Lantern_frames", (x + xx, y + yy, z - .2), (x + xx, y + yy, z + .2), .027, "iron")
        for zz in (-.2, .2):
            box("Lantern_frames", (x, y, z + zz), (.29, .29, .045), "iron")
        beam("Lantern_brackets", (x, y, z + .2), (x, y + .25, z + .59), .04, "iron")
        FEATURES["lanterns"] += 1


# The scene-composition entry point: reset repeatable choices, assemble all
# landmarks and off-camera context, then emit meshes and manifest audit counts.
def build(mats):
    global G
    RNG.seed(630117)
    FEATURES.clear()
    G = Geometry(mats)
    water_info = water()
    quay()
    house("Hero_Harbor_Inn", (13.0, 10.5, 1.15), 10.2, 9, 8.9, 7.6, .13, 0, True)
    house("Inn_rear_wing", (20.8, 15.2, 1.15), 5.6, 7.2, 7.4, 5.1, .10, 1)
    house("Shipwright_House", (5.2, 15.8, 1.15), 5.4, 6.4, 5.6, 4.6, -.07, 1)
    house("Green_Shutter_House", (-.4, 19.5, 1.15), 4.5, 5.8, 4.9, 3.2, 1.34, 2)
    house("Narrow_Merchant_House", (-5.1, 23.1, 1.15), 3.6, 5.5, 5.5, 3.3, -.12, 3)
    house("Old_Boathouse", (-9.3, 26.1, 1.15), 4.6, 6.1, 3.5, 2.8, .38, 4)
    house("Gabled_Store", (-13.7, 31.0, 1.15), 3.7, 5.1, 3.9, 3.0, -.08, 1)
    house("Distant_Ochre_House", (-17.8, 37.0, 1.15), 4.5, 5.8, 4.1, 3.2, .06, 3)
    house("Rear_Roofline", (1.2, 31.5, 1.15), 4.8, 6.6, 5.6, 4.0, .18, 2)
    tower()
    boat("Cream_Gaff_Cutter", (-9.0, 0.7, 0), 8.4, 3.35, -.96, 1, True, "canvas", True)
    boat("Distant_Cream_Sloop", (-5.6, 11.1, 0), 7.3, 2.6, -.31, .55, True, "canvas_weathered")
    boat("Green_Working_Sailboat", (1.7, 5.9, 0), 6.7, 2.2, -.45, .42, True, "canvas_teal")
    boat("Foreground_Rowboat", (6.9, -8.7, 0), 6.0, 2.05, -1.04, .57, False)
    boat("Tender_at_Cutter", (-12.4, -1.8, 0), 5.5, 1.8, -1.4, .40, False)
    boat("Village_Rowboat", (-3.6, 9.0, 0), 5.3, 1.8, -.3, .37, False)
    rope("Mooring_lines", (7.9, -8.0, .80), (10.3, -7.5, 1.75), .024, .3)
    rope("Mooring_lines", (1.9, 6.0, .75), (4.0, 6.8, 1.55), .018, .17)
    props()
    for name, pos, h, r, n in (
        ("Inn_front_tree", (19.1, 6.0, 1.2), 7.1, 2.25, 1250),
        ("Golden_tree_behind_inn", (7.7, 23, 1.2), 11.5, 3.15, 1350),
        ("Town_lime_tree", (-3.8, 29.5, 1.2), 9.4, 2.6, 1000),
        ("Tower_tree", (-9.5, 38, 1.2), 11.1, 3.2, 1000),
        ("Left_elm", (-17, 33.3, 1.2), 8.5, 2.8, 850),
        ("Back_garden_tree", (24.5, 17.0, 1.2), 9.7, 3.1, 950),
        ("Quay_shrub", (15.8, -5.1, 1.2), 1.55, .72, 190),
    ):
        tree(name, pos, h, r, n)
    for i in range(14):
        bird("Harbor_gull_" + str(i), (RNG.uniform(-15, 20), RNG.uniform(6, 35), RNG.uniform(13, 21)), RNG.uniform(.27, .52), RNG.random() * math.tau, i % 3 == 0)
    for i, pos in enumerate(((3.0, -10, .17), (4.0, -9.0, .17), (-5.0, -8.0, .18))):
        bird("Water_gull_" + str(i), pos, .28, i, True)
    # Low offshore island and rear trees continue beyond the framed facades.
    for i in range(7):
        ellipsoid("Distant_coastal_island", (-50 + i * 11, 145 + i * 4, -.2), (13, 17, 1.8 + .45 * (i % 3)), "distant_hill", 16, 7)
    objects, triangle_counts = G.emit()
    return objects, triangle_counts, dict(FEATURES), water_info
