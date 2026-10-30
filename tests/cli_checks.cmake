# Integration checks run the actual CLI in a generated directory with spaces.
# Only test copies/generated files are altered; shipped scene assets stay intact.
if(NOT DEFINED RENDERER OR NOT DEFINED TEST_DIR)
    message(FATAL_ERROR "RENDERER and TEST_DIR are required")
endif()
file(MAKE_DIRECTORY "${TEST_DIR}/working directory with spaces")
set(WORK "${TEST_DIR}/working directory with spaces")
set(SCENE "${WORK}/minimal scene.gltf")
# A triangle, named camera, and point light form a tiny self-contained data-URI
# fixture, so argument/output tests do not need Blender or a network connection.
file(WRITE "${SCENE}" [=[
{"asset":{"version":"2.0"},"extensionsUsed":["KHR_lights_punctual"],"extensions":{"KHR_lights_punctual":{"lights":[{"type":"point","intensity":20}]}},"buffers":[{"byteLength":36,"uri":"data:application/octet-stream;base64,AAAAwAAAAMAAAAAAAAAAQAAAAMAAAAAAAAAAAAAAAEAAAAAA"}],"bufferViews":[{"buffer":0,"byteLength":36}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[-2,-2,0],"max":[2,2,0]}],"materials":[{"pbrMetallicRoughness":{"metallicFactor":0}}],"meshes":[{"primitives":[{"attributes":{"POSITION":0},"material":0}]}],"cameras":[{"name":"Reference","type":"perspective","perspective":{"yfov":0.7,"znear":0.01}}],"nodes":[{"mesh":0},{"camera":0,"translation":[0,0,3]},{"translation":[0,2,2],"extensions":{"KHR_lights_punctual":{"light":0}}}],"scenes":[{"nodes":[0,1,2]}],"scene":0}
]=])
# One valid linear RGBE texel supplies an input whose hash can test overwrite safety.
string(ASCII 64 64 64 129 RGBE)
file(WRITE "${WORK}/sky.hdr" "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 1\n${RGBE}")
file(SHA256 "${WORK}/sky.hdr" SKY_BEFORE)
# A useful rejection means BOTH a nonzero exit and the program's error diagnostic;
# a crash or timeout alone must not count as a successful negative test.
function(expect_failure label)
    execute_process(COMMAND "${RENDERER}" ${ARGN} WORKING_DIRECTORY "${WORK}"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 30)
    if(result EQUAL 0)
        message(FATAL_ERROR "${label}: incorrectly succeeded\n${output}\n${error}")
    endif()
    if(NOT error MATCHES "Error:")
        message(FATAL_ERROR "${label}: no useful diagnostic\n${result}\n${output}\n${error}")
    endif()
    message(STATUS "PASS ${label}")
endfunction()
# Validate syntax, ranges, overflow, finite numbers, duplicates, and missing inputs.
expect_failure("missing scene")
expect_failure("missing value" --scene)
expect_failure("unknown option" --unknown)
expect_failure("bad preset" --preset high)
expect_failure("zero samples" --samples 0)
expect_failure("negative samples" --samples -3)
expect_failure("trailing integer garbage" --samples 12garbage)
expect_failure("overflow integer" --width 99999999999999999999999999999)
expect_failure("allocation product limit" --width 16384 --height 16384)
expect_failure("invalid threads" --threads -1)
expect_failure("excessive threads" --threads 257)
expect_failure("zero bounces" --bounces 0)
expect_failure("negative seed" --seed -1)
expect_failure("NaN exposure" --exposure nan)
expect_failure("infinite rotation" --environment-rotation inf)
expect_failure("negative sky intensity" --environment-intensity -1)
expect_failure("duplicate argument" --samples 1 --samples 2)
expect_failure("missing scene file" --scene does-not-exist.glb)
expect_failure("missing camera" --scene "${SCENE}" --camera Missing)
expect_failure("missing environment" --scene "${SCENE}" --environment missing.hdr)
# Inject unsupported features into generated JSON and require explicit rejection.
file(READ "${SCENE}" SIMPLE_JSON)
string(REPLACE "\"version\":\"2.0\"}," "\"version\":\"2.0\"},\"extensionsRequired\":[\"KHR_draco_mesh_compression\"]," UNSUPPORTED_JSON "${SIMPLE_JSON}")
file(WRITE "${WORK}/unsupported.gltf" "${UNSUPPORTED_JSON}")
expect_failure("unsupported required extension" --scene "${WORK}/unsupported.gltf")
string(REPLACE "\"metallicFactor\":0}}" "\"metallicFactor\":0},\"alphaMode\":\"BLEND\"}" BLEND_JSON "${SIMPLE_JSON}")
file(WRITE "${WORK}/blend.gltf" "${BLEND_JSON}")
expect_failure("unsupported alpha blending" --scene "${WORK}/blend.gltf")
# Copy the external-resource fixture before deliberately removing or targeting a map.
file(MAKE_DIRECTORY "${WORK}/external fixture")
file(COPY "${SOURCE_DIR}/scenes/fixtures/export_contract.gltf" "${SOURCE_DIR}/scenes/fixtures/export_contract.bin"
    "${SOURCE_DIR}/scenes/fixtures/textures" DESTINATION "${WORK}/external fixture")
set(EXTERNAL "${WORK}/external fixture/export_contract.gltf")
set(TEXTURE "${WORK}/external fixture/textures/uv_grid.png")
file(SHA256 "${TEXTURE}" TEXTURE_BEFORE)
# A rejected output path must leave the input texture byte-for-byte unchanged.
expect_failure("input texture overwrite" --scene "${EXTERNAL}" --output "${TEXTURE}")
file(SHA256 "${TEXTURE}" TEXTURE_AFTER)
if(NOT TEXTURE_BEFORE STREQUAL TEXTURE_AFTER)
    message(FATAL_ERROR "Overwrite protection modified an input texture")
endif()
file(REMOVE "${TEXTURE}")
expect_failure("missing external texture" --scene "${EXTERNAL}")
expect_failure("input overwrite through paired HDR" --scene "${SCENE}" --environment "${WORK}/sky.hdr" --output "${WORK}/sky.png")
# The derived .hdr path is protected too, even when --output names a .png file.
file(SHA256 "${WORK}/sky.hdr" SKY_AFTER)
if(NOT SKY_BEFORE STREQUAL SKY_AFTER)
    message(FATAL_ERROR "Overwrite protection modified the input HDR")
endif()
# These failures are portable and do not rely on administrator-only directories.
file(MAKE_DIRECTORY "${WORK}/directory.png")
expect_failure("unwritable output type" --scene "${SCENE}" --output "${WORK}/directory.png")
file(WRITE "${WORK}/parent-is-file" "not a directory")
expect_failure("unwritable output parent" --scene "${SCENE}" --output "${WORK}/parent-is-file/image.png")
# Finally prove success: explicit dimensions/samples precede --preset but must
# still win, resources resolve from another cwd, and both output formats exist.
execute_process(COMMAND "${RENDERER}" --scene "${SCENE}" --camera Reference --width 2 --height 1 --samples 1
    --preset final --threads 1 --bounces 2 --output "${WORK}/small output.png"
    WORKING_DIRECTORY "${TEST_DIR}" RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 30)
if(NOT result EQUAL 0 OR NOT output MATCHES "2x1, 1 spp" OR NOT EXISTS "${WORK}/small output.hdr")
    message(FATAL_ERROR "Explicit preset overrides or paths with spaces/nondefault cwd failed: ${result}\n${output}\n${error}")
endif()
message(STATUS "PASS explicit preset overrides, local data URI, spaces, nondefault cwd, and output pair")
