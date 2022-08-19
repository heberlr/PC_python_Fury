# PhysiCell-Fury

This [Fury](https://fury.gl/) viewer is designed to visualize the outputs from of the PhysiCell 3-D model. In this release, we also use [PyVista](https://docs.pyvista.org/) to visualize substrate surfaces.

You can install Fury and PyVista using 'pip' comand:

```bash
pip install fury pyvista
```

> The python script has two options to run:
>
> > **Option 1 - Mode snapshot:**: to take 3D snapshots from the all frames in the source folder (useful for making movies).

```bash
python fury3D.py path_source_folder
```

> > **Option 2 - Mode interactive:** to interact with the scene.

```bash
python fury3D.py path_source_folder frame_ID
```

### Mode interactive

In the mode interact, when the Fury window appears, you can see the menu options:

- **Substrates**: select the substrate to visualize in Pyvista.
- **Cutting plane XY**: define the range on the z-axis to cut.
- **Cutting plane XZ**: define the range on the y-axis to cut
- **Cutting plane YZ**: define the range on the x-axis to cut
- **Reset Camera**: reset camera to default position.
- **Reset**: reset the cuties and show all the cells.
- **Snapshot**: save a Snapshot\_\*.jpg file in the source folder

Besides the menu options, you can interact with the scene using your mouse or trackpad

- _Rotation_: hold the left button of the mouse and drag.
- _Move camera_: hold shift + the left button of the mouse and drag.
- _Zoom_: hold right button of the mouse and drag or use the scroll button of the mouse.

###Example:

```bash
python fury3D.py output_immune 0
```

### Custom example:

The script `My_fury3D.py` is a example of customization of `coloring_function` and `header_function`. To visualize this example use:

```bash
python My_fury3D.py output_metastasis 27
```
