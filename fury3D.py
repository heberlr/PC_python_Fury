# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 18:00:01 2020

@author: fkurtog
"""

from pyMCDS import pyMCDS
import numpy as np
import pandas as pd
from fury import window, actor, ui, convert
import pyvista as pv
import glob, os
import sys

def coloring_function_default(df_cells):
    Cell_types = df_cells['cell_type'].unique()
    # Default colors (default color will be grey )
    Colors_default = np.array([[255,87,51], [255,0,0], [255,211,0], [0,255,0], [0,0,255], [254,51,139], [255,131,0], [50,205,50], [0,255,255], [255,0,127], [255,218,185], [143,188,143], [135,206,250]])/255.0 # grey, red, yellow, green, blue, magenta, orange, lime, cyan, hotpink, peachpuff, darkseagreen, lightskyblue
    apoptotic_color = np.array([255,255,255])/255.0 # white
    necrotic_color = np.array([139,69,19])/255.0 # brown
    # Colors
    Colors = np.ones((len(df_cells),4)) # 4 channels RGBO
    for index,type in enumerate(Cell_types):
        idxs = df_cells.loc[ (df_cells['cell_type'] == type) & (df_cells['cycle_model'] < 100) ].index
        Colors[idxs,0] = Colors_default[index,0]
        Colors[idxs,1] = Colors_default[index,1]
        Colors[idxs,2] = Colors_default[index,2]
        idxs_apoptotic = df_cells.loc[df_cells['cycle_model'] == 100].index
        Colors[idxs_apoptotic,0] = apoptotic_color[0]
        Colors[idxs_apoptotic,1] = apoptotic_color[1]
        Colors[idxs_apoptotic,2] = apoptotic_color[2]
        idxs_necrotic = df_cells.loc[df_cells['cycle_model'] > 100].index
        Colors[idxs_necrotic,0] = necrotic_color[0]
        Colors[idxs_necrotic,1] = necrotic_color[1]
        Colors[idxs_necrotic,2] = necrotic_color[2]
    return Colors

def header_function_default(mcds):
    # Current time
    curr_time = mcds.get_time() # min
    time_days = curr_time//1440.0
    time_hours = (curr_time%1440.0)//60
    time_min = ((curr_time%1440.0)%60)
    # Number of cells
    Num_cells = len(mcds.data['discrete_cells']['ID'])
    title_text = "Current time: %02d days, %02d hours, and %0.2f minutes, %d agents"%(time_days,time_hours,time_min,Num_cells)
    return title_text

def CreateScene(folder, InputFile, coloring_function = coloring_function_default, header_function = header_function_default, clipping_quarter=True, SaveImage=False):
    # Reading data
    mcds=pyMCDS(InputFile,folder)
    # Define domain size
    centers = mcds.get_linear_voxels()
    X = np.unique(centers[0, :])
    Y = np.unique(centers[1, :])
    Z = np.unique(centers[2, :])
    dx = (X.max() - X.min()) / (X.shape[0]-1)
    dy = (Y.max() - Y.min()) / (Y.shape[0]-1)
    dz = (Z.max() - Z.min()) / (Z.shape[0]-1)
    x_min_domain = round(mcds.data['mesh']['x_coordinates'][0,0,0]-0.5*dx)
    y_min_domain = round(mcds.data['mesh']['y_coordinates'][0,0,0]-0.5*dy)
    z_min_domain = round(mcds.data['mesh']['z_coordinates'][0,0,0]-0.5*dz)
    x_max_domain = round(mcds.data['mesh']['x_coordinates'][-1,-1,-1]+0.5*dx)
    y_max_domain = round(mcds.data['mesh']['y_coordinates'][-1,-1,-1]+0.5*dy)
    z_max_domain = round(mcds.data['mesh']['z_coordinates'][-1,-1,-1]+0.5*dz)
    # Cell positions
    ncells = len(mcds.data['discrete_cells']['ID'])
    C_xyz = np.zeros((ncells,3))
    C_xyz[:,0] =  mcds.data['discrete_cells']['position_x']
    C_xyz[:,1] =  mcds.data['discrete_cells']['position_y']
    C_xyz[:,2] =  mcds.data['discrete_cells']['position_z']
    # Cell Radius Calculation
    C_radii = np.cbrt(mcds.data['discrete_cells']['total_volume'] * 0.75 / np.pi) # r = np.cbrt(V * 0.75 / pi)
    # Coloring
    C_colors = coloring_function(mcds.get_cell_df())
    # if ( clipping_quarter ):
    #     # x > 0 or y > 0
    #     idx_hiddencells = np.argwhere( (C_xyz[:,0] > 0) & (C_xyz[:,1] > 0) ).flatten()
    #     C_colors[idx_hiddencells,3] = 0 # opacity = 0

    ###############################################################################################
    # Creaating Scene
    size_window = (1000,1000)
    showm = window.ShowManager(size=size_window, reset_camera=True, order_transparent=True)
    ###############################################################################################
    # TITLE
    title_text = header_function(mcds)
    title = ui.TextBlock2D(text=title_text, font_size=20, font_family='Arial', justification='center', vertical_justification='bottom', bold=False, italic=False, shadow=False, color=(1, 1, 1), bg_color=None, position=(500, 900))
    showm.scene.add(title)
    # Drawing Domain Boundaries
    lines = [np.array([[x_min_domain,y_min_domain,z_min_domain],[x_max_domain,y_min_domain,z_min_domain],[x_max_domain,y_max_domain,z_min_domain],[x_min_domain,y_max_domain,z_min_domain],[x_min_domain,y_min_domain,z_min_domain],[x_min_domain,y_min_domain,z_max_domain],[x_min_domain,y_max_domain,z_max_domain],[x_min_domain,y_max_domain,z_min_domain],[x_min_domain,y_max_domain,z_max_domain],[x_max_domain,y_max_domain,z_max_domain],[x_max_domain,y_max_domain,z_min_domain],[x_max_domain,y_max_domain,z_min_domain],[x_max_domain,y_min_domain,z_min_domain],[x_max_domain,y_min_domain,z_max_domain],[x_max_domain,y_max_domain,z_max_domain],[x_max_domain,y_min_domain,z_max_domain],[x_min_domain,y_min_domain,z_max_domain]])]
    colors = np.array([0.5, 0.5, 0.5]) # Gray
    domain_box = actor.line(lines, colors)
    showm.scene.add(domain_box)
    ###############################################################################################
    # Planes sections
    center = np.array([[0,0,0]])
    box_actorXY = actor.box(center, np.array([[1,1,0]]), colors=(0.5, 0.5, 0.5,0.4),scales=(2*y_max_domain,2*x_max_domain,0.5*dz))
    box_actorXZ = actor.box(center, np.array([[1,0,1]]), colors=(0.5, 0.5, 0.5,0.4),scales=(2*x_max_domain,0.5*dy,2*z_max_domain))
    box_actorYZ = actor.box(center, np.array([[0,1,1]]), colors=(0.5, 0.5, 0.5,0.4),scales=(0.5*dx,2*z_max_domain,2*y_max_domain))
    showm.scene.add(box_actorXY)
    showm.scene.add(box_actorXZ)
    showm.scene.add(box_actorYZ)
    # Section in plane XY
    line_slider_xy = ui.LineSlider2D(center=(150, 50), initial_value=0, min_value=z_min_domain, max_value=z_max_domain, orientation="horizontal")
    showm.scene.add(line_slider_xy)
    def translate_planeXY(slider):
        shift = np.array([0,0,slider.value])
        center = np.zeros(3)
        box_actorXY.SetPosition(center+shift)
    line_slider_xy.on_change = translate_planeXY
    line_slider_xy_label = ui.TextBlock2D(position=(260,40),text="plane XY")
    showm.scene.add(line_slider_xy_label)
    # Section in plane XZ
    line_slider_xz = ui.LineSlider2D(center=(150, 100), initial_value=0, min_value=y_min_domain, max_value=y_max_domain, orientation="horizontal")
    showm.scene.add(line_slider_xz)
    def translate_planeXZ(slider):
        shift = np.array([0,slider.value,0])
        center = np.zeros(3)
        box_actorXZ.SetPosition(center+shift)
    line_slider_xz.on_change = translate_planeXZ
    line_slider_xz_label = ui.TextBlock2D(position=(260,90),text="plane XZ")
    showm.scene.add(line_slider_xz_label)
    # Section in plane YZ
    line_slider_yz = ui.LineSlider2D(center=(150, 150), initial_value=0, min_value=x_min_domain, max_value=x_max_domain, orientation="horizontal")
    line_slider_yz.default_color = (1,0,0) # Red
    showm.scene.add(line_slider_yz)
    def translate_planeYZ(slider):
        shift = np.array([slider.value,0,0])
        center = np.zeros(3)
        box_actorYZ.SetPosition(center+shift)
    line_slider_yz.on_change = translate_planeYZ
    line_slider_yz_label = ui.TextBlock2D(position=(260,140),text="plane YZ")
    showm.scene.add(line_slider_yz_label)
    ###############################################################################################
    # Add referencial vectors axis
    center = np.array([[x_max_domain,y_min_domain,z_max_domain],[x_max_domain,y_min_domain,z_max_domain],[x_max_domain,y_min_domain,z_max_domain]])
    direction_x = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
    arrow_actor = actor.arrow(center,direction_x,np.array([[1,0,0],[0,1,0],[0,0,1]]),heights=0.5*min(x_max_domain,y_max_domain,z_max_domain),tip_radius=0.1)
    showm.scene.add(arrow_actor)
    ###############################################################################################
    # Creating Sphere Actor for all cells
    sphere_actor = actor.sphere(centers=C_xyz,colors=C_colors,radii=C_radii)
    showm.scene.add(sphere_actor)
    ###############################################################################################
    # Substrates
    substrates = mcds.get_substrate_names()
    substrate_combobox = ui.ComboBox2D(items=substrates, placeholder="Choose substrate", position=(150, 50), size=(500, 100))
    # Preparing pyvista mesh
    substrate0 = np.transpose(mcds.get_concentrations(substrates[0]), (1,0,2)) # CHECK ORDER ON PYMCDS
    # Create the spatial reference
    grid = pv.UniformGrid()
    # Set the grid dimensions: shape + 1 because we want to inject our values on
    # the CELL data (mesh)
    grid.dimensions = np.array(substrate0.shape) + 1
    # Edit the spatial reference
    grid.origin = (x_min_domain, y_min_domain, z_min_domain)  # The bottom left corner of the data set
    grid.spacing = (dx, dy, dz)  # These are the cell sizes along each axis
    # Add the data values to the cell data
    grid.cell_data[str(substrates[0])] = substrate0.flatten(order="F")  # Flatten the array!
    pv.global_theme.cmap = 'coolwarm_r' # cmap palette color
    # Add others Substrates
    for i in range(1,len(substrates)):
        grid.cell_data.set_array( np.transpose(mcds.get_concentrations(substrates[i]), (1,0,2)).flatten(order="F"),substrates[i]) # CHECK ORDER ON PYMCDS
    def change_substrate(combobox):
        selected_substrate = combobox.selected_text
        grid.cell_data.active_scalars_name = selected_substrate # select substrate on grid
        # p = grid.plot(show_edges=True)
        p = pv.Plotter()
        p.add_text(selected_substrate)
        boxwidgets = p.add_mesh_clip_box(grid)
        # print(boxwidgets.box_clipped_meshes)
        p.add_camera_orientation_widget()
        #print(p.box_clipped_meshes)
        p.show()
        # print(grid)
    substrate_combobox.on_change = change_substrate
    showm.scene.add(substrate_combobox)
    ###############################################################################################
    # Menu list
    MenuValues = ['Substrates','Cutting plane XY','Cutting plane XZ','Cutting plane YZ','Reset Camera']
    Actors = [[substrate_combobox],[line_slider_xy,line_slider_xy_label],[line_slider_xz,line_slider_xz_label],[line_slider_yz,line_slider_yz_label]]
    listbox = ui.ListBox2D(values=MenuValues, position=(700, 0), size=(300, 200), multiselection=False)
    def hide_all_widgets():
        for actor  in Actors:
            for element in actor:
                element.set_visibility(False)
        box_actorXY.SetVisibility(False)
        box_actorXZ.SetVisibility(False)
        box_actorYZ.SetVisibility(False)
    hide_all_widgets()
    def MenuOption():
        hide_all_widgets()
        if MenuValues[MenuValues.index(listbox.selected[0])] == 'Substrates':
            substrate_combobox.set_visibility(True)
        if MenuValues[MenuValues.index(listbox.selected[0])] == 'Cutting plane XY':
            box_actorXY.SetVisibility(True)
            line_slider_xy.set_visibility(True)
            line_slider_xy_label.set_visibility(True)
        if MenuValues[MenuValues.index(listbox.selected[0])] == 'Cutting plane XZ':
            box_actorXZ.SetVisibility(True)
            line_slider_xz.set_visibility(True)
            line_slider_xz_label.set_visibility(True)
        if MenuValues[MenuValues.index(listbox.selected[0])] == 'Cutting plane YZ':
            box_actorYZ.SetVisibility(True)
            line_slider_yz.set_visibility(True)
            line_slider_yz_label.set_visibility(True)
        if MenuValues[MenuValues.index(listbox.selected[0])] == 'Reset Camera':
            showm.scene.set_camera(position=(2.75*x_min_domain, 0, 7.0*z_max_domain), focal_point=(0, 0, 0), view_up=(0, 0, 0))
    listbox.on_change = MenuOption
    showm.scene.add(listbox)
    ###############################################################################################
    # Show Manager
    showm.scene.reset_camera()
    showm.scene.set_camera(position=(2.75*x_min_domain, 0, 7.0*z_max_domain), focal_point=(0, 0, 0), view_up=(0, 0, 0))
    ###############################################################################################
    # Save image
    if ( SaveImage ):
        window.record(showm.scene,size=size_window,out_path=folder+os.path.splitext(InputFile)[0]+".jpg")
    else:
        showm.start()

def CreateSnapshots(folder, coloring_function = coloring_function_default, header_function = header_function_default):
    files = glob.glob(folder+'out*.xml')
    # Make snapshots
    for file in files:
        CreateScene(folder,os.path.basename(file),coloring_function=coloring_function_default, header_function=header_function_default,SaveImage=True)

if __name__ == '__main__':
    if (len(sys.argv) != 3 and len(sys.argv) != 2):
      print("Please provide\n 1 arg [folder]: to taking snapshots from the folder \n or provide 2 args [folder] [frame ID]: to interact with scene!")
      sys.exit(1)
    if (len(sys.argv) == 3):
      CreateScene(sys.argv[1],"output%08d.xml"%int(sys.argv[2]))
    if (len(sys.argv) == 2):
      CreateSnapshots(sys.argv[1])
