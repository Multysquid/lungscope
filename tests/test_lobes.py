"""The lung from lobe masks, through every layer: reading the masks and lobes.json, fitting where
they sit on a tree, their surfaces in the scene, and the app's --lobes and lungscope-fit-lobes.

The masks are two ellipsoids (conftest.lobe_masks) in NIfTI files written by hand.
"""

import json
import sys

import numpy as np
import pytest
import pyvista as pv
from conftest import WORLD_FROM_INDEX, lobe_masks, write_nifti
from trame.app import get_server

from lungscope import app
from lungscope.analysis import fit_lobes, terminal_unit_centres
from lungscope.data import load_lobes, read_input, save_lobe_transform
from lungscope.data.lobes import MARGIN, read_mask
from lungscope.views import LungScene, lobe_surface

# a known transform from image to mesh coordinates: axes swapped, turned 10 degrees, shrunk to 0.4
_TURN = np.radians(10.0)
_ROTATION = np.array([[np.cos(_TURN), -np.sin(_TURN), 0.0], [np.sin(_TURN), np.cos(_TURN), 0.0], [0.0, 0.0, 1.0]])
_SWAP = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
MESH_FROM_WORLD = np.eye(4)
MESH_FROM_WORLD[:3, :3] = 0.4 * _ROTATION @ _SWAP
MESH_FROM_WORLD[:3, 3] = [200.0, 50.0, 210.0]


def apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=float) @ matrix[:3, :3].T + matrix[:3, 3]


def units_inside(count: int, seed: int = 1) -> np.ndarray:
    """Image coordinates of ``count`` points spread over the lobes."""
    rng = np.random.default_rng(seed)
    index = np.argwhere(np.logical_or(*lobe_masks()))
    picked = index[rng.choice(len(index), count, replace=False)] + rng.uniform(-0.3, 0.3, (count, 3))
    return apply(WORLD_FROM_INDEX, picked)


# data


@pytest.mark.parametrize(
    "world_from_index",
    [WORLD_FROM_INDEX, np.diag([-0.7, 0.7, 1.25, 1.0])],  # the second one mirrors (negative determinant)
    ids=["scanner", "mirrored"],
)
def test_a_mask_is_cropped_and_keeps_its_image_coordinates(tmp_path, world_from_index):
    mask = np.zeros((30, 20, 40), dtype=bool)
    mask[10:14, 1:4, 20:23] = True
    mask[12, 3, 24] = True
    write_nifti(tmp_path / "lobe_0.nii.gz", mask, world_from_index)
    lobe = read_mask(tmp_path / "lobe_0.nii.gz")
    assert lobe.name == "lobe_0"
    assert lobe.mask.shape == (4 + 2 * MARGIN, 3 + 1 + MARGIN, 5 + 2 * MARGIN)  # j stops at the image's edge
    assert lobe.mask.sum() == mask.sum()
    # the same voxel positions; VTK may store the slices of a mirrored image in reverse order
    expected = apply(world_from_index, np.argwhere(mask))
    found = apply(lobe.world_from_index, np.argwhere(lobe.mask))
    np.testing.assert_allclose(found[np.lexsort(found.T)], expected[np.lexsort(expected.T)], atol=1e-5)


def test_lobes_need_their_lobes_json_which_save_writes(lobe_folder):
    with pytest.raises(FileNotFoundError, match="lobes.json is missing"):
        load_lobes(lobe_folder)
    lobes = load_lobes(lobe_folder, placed=False)
    assert [lobe.name for lobe in lobes.lobes] == ["lobe_0", "lobe_1"]
    assert lobes.mesh_from_world is None
    with pytest.raises(ValueError, match="no lobes.json"):
        lobes.mesh_from_index(lobes.lobes[0])

    save_lobe_transform(lobe_folder, MESH_FROM_WORLD, {"method": "known"})
    lobes = load_lobes(lobe_folder)
    np.testing.assert_allclose(lobes.mesh_from_world, MESH_FROM_WORLD)
    assert lobes.fit == {"method": "known"}
    for lobe, mask in zip(lobes.lobes, lobe_masks(), strict=True):
        found = apply(lobes.mesh_from_index(lobe), np.argwhere(lobe.mask))
        np.testing.assert_allclose(found, apply(MESH_FROM_WORLD @ WORLD_FROM_INDEX, np.argwhere(mask)), atol=1e-4)


@pytest.mark.parametrize(
    "content, message",
    [
        ({"fit": {}}, "needs mesh_from_world"),
        ({"mesh_from_world": np.eye(3).tolist()}, "last row"),
        ({"mesh_from_world": np.diag([1.0, 0.0, 1.0, 1.0]).tolist()}, "not invertible"),
    ],
)
def test_a_lobes_json_that_is_no_affine_map_is_refused(lobe_folder, content, message):
    (lobe_folder / "lobes.json").write_text(json.dumps(content))
    with pytest.raises(ValueError, match=message):
        load_lobes(lobe_folder)


def test_a_folder_without_masks_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="no lobe masks"):
        load_lobes(tmp_path, placed=False)


# analysis


def test_terminal_unit_centres_are_the_midpoints_of_elements_with_volume():
    ends = np.array([[[0, 0, 0], [0, 0, 2]], [[0, 0, 2], [2, 0, 2]], [[0, 0, 2], [0, 4, 2]]], dtype=float)
    centres = terminal_unit_centres(ends, np.array([0.0, 5.0, np.nan]))
    np.testing.assert_allclose(centres, [[1, 0, 2]])


def test_the_fit_finds_the_transform_that_puts_the_units_into_the_lobes():
    world = units_inside(2000)
    units = apply(MESH_FROM_WORLD, world)
    masks = lobe_masks()
    fit = fit_lobes(units, masks, [WORLD_FROM_INDEX] * 2)  # on the masks coarsened to 3 to 4 mm voxels
    assert fit.inside > 0.97
    assert fit.similarity_inside > 0.97
    assert sum(fit.per_lobe) == round(fit.inside * len(units))
    assert fit.per_lobe[0] > fit.per_lobe[1]  # the big lobe holds most units
    back = apply(np.linalg.inv(fit.mesh_from_world), units)
    assert np.median(np.linalg.norm(back - world, axis=1)) < 1.5  # mm


# views


@pytest.mark.parametrize("mirror", [1.0, -1.0])
def test_lobe_surfaces_are_closed_placed_and_face_outwards(mirror):
    mesh_from_index = np.diag([0.5, 0.5 * mirror, 0.5, 1.0])
    mesh_from_index[:3, 3] = [10.0, 20.0, 30.0]
    lung = lobe_surface(lobe_masks(), [mesh_from_index] * 2)
    assert lung.n_cells > 0 and lung.n_open_edges == 0
    expected = apply(mesh_from_index, np.argwhere(np.logical_or(*lobe_masks())))
    bounds = np.reshape(lung.bounds, (3, 2))
    np.testing.assert_allclose(bounds[:, 0], expected.min(axis=0), atol=1.0)  # within two voxels
    np.testing.assert_allclose(bounds[:, 1], expected.max(axis=0), atol=1.0)
    big = lobe_surface(lobe_masks()[:1], [mesh_from_index])  # an ellipsoid: every normal points away from its centre
    outward = np.einsum("ij,ij->i", big.point_normals, big.points - big.points.mean(axis=0))
    assert np.all(outward > 0)


def test_empty_masks_give_no_lung():
    assert lobe_surface([np.zeros((4, 4, 4), dtype=bool)], [np.eye(4)]) is None


def test_the_scene_draws_the_lung_it_is_given():
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -10.0], [0.0, 0.0, -15.0]])
    tree = pv.UnstructuredGrid(np.array([2, 0, 1, 2, 1, 2]), np.full(2, pv.CellType.LINE, dtype=np.uint8), points)
    tree.cell_data["radius"] = [2.0, 0.0]
    tree.cell_data["v0"] = [0.0, 1000.0]
    lung = pv.Sphere(radius=20.0, center=(0.0, 0.0, -8.0))
    scene = LungScene(tree, lung=lung)
    assert scene.has_lung
    np.testing.assert_allclose(scene.lung_actor.mapper.dataset.bounds, lung.bounds)
    assert scene.lung_actor.GetPickable() == 0


# app


@pytest.fixture
def unit_mesh(tmp_path):
    """A mesh of 600 terminal units (short line elements with v0) inside the lobes, in mesh
    coordinates, and the units' image coordinates."""
    world = units_inside(600, seed=2)
    centres = apply(MESH_FROM_WORLD, world)
    points = np.repeat(centres, 2, axis=0) + np.tile([[-0.1, 0, 0], [0.1, 0, 0]], (len(centres), 1))
    cells = np.column_stack([np.full(len(centres), 2), np.arange(0, 2 * len(centres), 2), np.arange(1, 2 * len(centres), 2)])
    mesh = pv.UnstructuredGrid(cells.ravel(), np.full(len(centres), pv.CellType.LINE, dtype=np.uint8), points)
    mesh.cell_data["v0"] = np.ones(len(centres))
    path = tmp_path / "units.vtu"
    mesh.save(path)
    return path, world


def test_fit_lobes_writes_lobes_json_and_keeps_an_existing_one(lobe_folder, unit_mesh, monkeypatch, capsys):
    mesh, world = unit_mesh
    monkeypatch.setattr(sys, "argv", ["lungscope-fit-lobes", str(lobe_folder), "--mesh", str(mesh)])
    app.fit_lobes_main()
    assert "Wrote" in capsys.readouterr().out
    lobes = load_lobes(lobe_folder)
    assert lobes.fit["terminal_units"] == 600
    assert lobes.fit["mesh"] == "../units.vtu"
    assert lobes.fit["inside"] > 0.9
    assert set(lobes.fit["units_per_lobe"]) == {"lobe_0", "lobe_1"}
    ends, arrays = read_input(mesh)
    back = apply(np.linalg.inv(lobes.mesh_from_world), terminal_unit_centres(ends, arrays["v0"]))
    assert np.median(np.linalg.norm(back - world, axis=1)) < 2.0

    with pytest.raises(SystemExit) as stop:  # a second fit would replace it
        app.fit_lobes_main()
    assert stop.value.code == 2
    assert "already has a lobes.json; --force replaces it" in capsys.readouterr().err


def fresh_main(monkeypatch, name: str, arguments: str):
    """Run app.main with its own server, which does not start; returns the server and the scene."""
    server = get_server(name, client_type="vue3")
    server.start = lambda *_, **__: None
    monkeypatch.setattr(app, "get_server", lambda **_: server)
    monkeypatch.setenv("TRAME_ARGS", arguments)
    built = {}
    original = app.build_viewer

    def build_viewer(server, scene, title):
        built["scene"] = scene
        return original(server, scene, title)

    monkeypatch.setattr(app, "build_viewer", build_viewer)
    app.main()
    return server, built.get("scene")


def test_lobes_need_a_mesh_or_a_study(monkeypatch, capsys, lobe_folder):
    with pytest.raises(SystemExit) as stop:
        fresh_main(monkeypatch, "test_lobes_alone", f"--lobes {lobe_folder} --server --port 0")
    assert stop.value.code == 2
    assert "--lobes needs --study or --mesh" in capsys.readouterr().err


def test_lobes_without_lobes_json_are_refused_at_once(monkeypatch, capsys, lobe_folder, unit_mesh):
    with pytest.raises(SystemExit) as stop:
        fresh_main(monkeypatch, "test_lobes_unplaced", f"--mesh {unit_mesh[0]} --lobes {lobe_folder} --server")
    assert stop.value.code == 2
    assert "lobes.json is missing" in capsys.readouterr().err


def test_the_viewer_draws_the_lung_from_the_lobes(monkeypatch, lobe_folder, unit_mesh):
    save_lobe_transform(lobe_folder, MESH_FROM_WORLD, {"method": "known"})
    server, scene = fresh_main(monkeypatch, "test_lobes_viewer", f"--mesh {unit_mesh[0]} --lobes {lobe_folder} --server")
    assert server.state.has_lung and scene.has_lung
    expected = apply(MESH_FROM_WORLD @ WORLD_FROM_INDEX, np.argwhere(np.logical_or(*lobe_masks())))
    bounds = np.reshape(scene.lung_actor.mapper.dataset.bounds, (3, 2))
    np.testing.assert_allclose(bounds[:, 0], expected.min(axis=0), atol=1.5)
    np.testing.assert_allclose(bounds[:, 1], expected.max(axis=0), atol=1.5)
