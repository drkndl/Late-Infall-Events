from paraview.simple import *

# Opening the file
reader = OpenDataFile("/Users/drishikanadella/Desktop/gasdens450.vtk")

# Applying the CleantoGrid filter
clean = CleantoGrid(Input=reader)
clean.UpdatePipeline()

# Calculator filter
calc = Calculator(Input=clean)
calc.CoordinateResults = 1

calc.ResultArrayName = "cartesian_densities"
calc.Function = (
    "coordsX*sin(coordsY)*cos(coordsZ)*iHat + "
    "coordsX*sin(coordsY)*sin(coordsZ)*jHat + "
    "coordsX*cos(coordsY)*kHat"
)

# Apply
calc.UpdatePipeline()

# Applying the calculator to convert the coordinates from spherical to Cartesian
Show(calc)