from paraview.simple import *
import numpy as np 
import os

print("I am in thesis/paraview_python")
print("Current working directory:", os.getcwd())

# Opening the file
reader = OpenDataFile("/Users/drishikanadella/Desktop/gasdens450.vtk")

# Applying the CleantoGrid filter
clean = CleantoGrid(Input=reader)
clean.UpdatePipeline()

# Calculator filter
calc = Calculator(Input=clean)
calc.CoordinateResults = 1

# Converting sphereical to Cartesian densities
calc.ResultArrayName = "cartesian_densities"
calc.Function = (
    "coordsX*sin(coordsY)*cos(coordsZ)*iHat + "
    "coordsX*sin(coordsY)*sin(coordsZ)*jHat + "
    "coordsX*cos(coordsY)*kHat"
)
calc.UpdatePipeline()
# Show(calc)

# Generating isosurfaces
cont_array = np.logspace(-18, -11, 30, base=10.)
dens_contour = Contour(Input=calc, Isosurfaces=cont_array) #ContourBy="dens"
dens_contour.UpdatePipeline()

# Change contour opacity and colour map properties 
disp = Show(dens_contour)
ColorBy(disp, ('gasdens'))
colorMap = GetColorTransferFunction('gasdens')
colorMap.ApplyPreset('Cool to Warm (Extended)')
# disp.RescaleTransferFunctionToDataRange(True)       # If you want the colormap to match the full contour range cont_array
colorMap.RescaleTransferFunction(10**-20, 10**-11)
colorMap.UseLogScale = 1
disp.Opacity = 0.25                               
disp.SetScalarBarVisibility(GetActiveView(), False)    # Display colour bar

# Colour bar properties
# view = GetActiveViewOrCreate('RenderView')
scalarBar = GetScalarBar(colorMap) #, view)
scalarBar.ScalarBarLength = 0.01 
scalarBar.ScalarBarThickness = 30  
scalarBar.Title = "Gas density (g/cm^3)"

# Show the disk(s)
Render()

# Save image
mydisks = GetActiveView()
SaveScreenshot("/Users/drishikanadella/Desktop/final_state.png", mydisks, ImageResolution=[1920, 1080], OverrideColorPalette="Black Background")

# SaveAnimation('animation.avi', GetActiveView(), FrameWindow = [1, 100], FrameRate = 1)