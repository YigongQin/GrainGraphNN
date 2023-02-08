# trace generated using paraview version 5.11.0
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
import sys

directoryOut = '/Users/yigongqin/Documents/Research/ML/Grain/visualization/'
directoryOut = '.'
datasetIn = sys.argv[1] if len(sys.argv)>1 else directoryOut+'test.vtk'
directoryOut = sys.argv[2] if len(sys.argv)>2 else directoryOut
imageFilesOut = sys.argv[3] if len(sys.argv)>3 else 'grain'
print("datasetIn = " + datasetIn)
print("directoryOut = " + directoryOut)
print("imageFilesOut = " + imageFilesOut)


#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'Legacy VTK Reader'
seed20_G50_R50vtk = LegacyVTKReader(registrationName='test.vtk', FileNames=[datasetIn])

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
seed20_G50_R50vtkDisplay = Show(seed20_G50_R50vtk, renderView1, 'UniformGridRepresentation')

# trace defaults for the display properties.
seed20_G50_R50vtkDisplay.Representation = 'Outline'
seed20_G50_R50vtkDisplay.ColorArrayName = ['POINTS', '']
seed20_G50_R50vtkDisplay.SelectTCoordArray = 'None'
seed20_G50_R50vtkDisplay.SelectNormalArray = 'None'
seed20_G50_R50vtkDisplay.SelectTangentArray = 'None'
seed20_G50_R50vtkDisplay.OSPRayScaleArray = 'theta_z'
seed20_G50_R50vtkDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
seed20_G50_R50vtkDisplay.SelectOrientationVectors = 'None'
seed20_G50_R50vtkDisplay.ScaleFactor = 2.0
seed20_G50_R50vtkDisplay.SelectScaleArray = 'theta_z'
seed20_G50_R50vtkDisplay.GlyphType = 'Arrow'
seed20_G50_R50vtkDisplay.GlyphTableIndexArray = 'theta_z'
seed20_G50_R50vtkDisplay.GaussianRadius = 0.1
seed20_G50_R50vtkDisplay.SetScaleArray = ['POINTS', 'theta_z']
seed20_G50_R50vtkDisplay.ScaleTransferFunction = 'PiecewiseFunction'
seed20_G50_R50vtkDisplay.OpacityArray = ['POINTS', 'theta_z']
seed20_G50_R50vtkDisplay.OpacityTransferFunction = 'PiecewiseFunction'
seed20_G50_R50vtkDisplay.DataAxesGrid = 'GridAxesRepresentation'
seed20_G50_R50vtkDisplay.PolarAxes = 'PolarAxesRepresentation'
seed20_G50_R50vtkDisplay.ScalarOpacityUnitDistance = 0.1385685404161374
seed20_G50_R50vtkDisplay.OpacityArrayName = ['POINTS', 'theta_z']
seed20_G50_R50vtkDisplay.IsosurfaceValues = [42.154137273875]
seed20_G50_R50vtkDisplay.SliceFunction = 'Plane'
seed20_G50_R50vtkDisplay.Slice = 123

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
seed20_G50_R50vtkDisplay.ScaleTransferFunction.Points = [0.75957012975, 0.0, 0.5, 0.0, 83.548704418, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
seed20_G50_R50vtkDisplay.OpacityTransferFunction.Points = [0.75957012975, 0.0, 0.5, 0.0, 83.548704418, 1.0, 0.5, 0.0]

# init the 'Plane' selected for 'SliceFunction'
seed20_G50_R50vtkDisplay.SliceFunction.Origin = [10.0, 10.0, 9.88]

# reset view to fit data
renderView1.ResetCamera(False)

# get the material library
materialLibrary1 = GetMaterialLibrary()

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(seed20_G50_R50vtkDisplay, ('POINTS', 'theta_z'))

# rescale color and/or opacity maps used to include current data range
seed20_G50_R50vtkDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
seed20_G50_R50vtkDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'theta_z'
theta_zLUT = GetColorTransferFunction('theta_z')

# get opacity transfer function/opacity map for 'theta_z'
theta_zPWF = GetOpacityTransferFunction('theta_z')

# change representation type
seed20_G50_R50vtkDisplay.SetRepresentationType('Surface')

# Properties modified on renderView1
renderView1.UseColorPaletteForBackground = 0

# Properties modified on renderView1
renderView1.Background = [1.0, 1.0, 1.0]

# hide color bar/color legend
seed20_G50_R50vtkDisplay.SetScalarBarVisibility(renderView1, False)

# show color bar/color legend
seed20_G50_R50vtkDisplay.SetScalarBarVisibility(renderView1, True)

# get color legend/bar for theta_zLUT in view renderView1
theta_zLUTColorBar = GetScalarBar(theta_zLUT, renderView1)

# Properties modified on theta_zLUTColorBar
theta_zLUTColorBar.Title = '$\\theta_z$'
theta_zLUTColorBar.UseCustomLabels = 1
theta_zLUTColorBar.CustomLabels = [30.0, 60.0]
theta_zLUTColorBar.AddRangeLabels = 0
theta_zLUTColorBar.ScalarBarThickness = 32

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(1165, 804)

# current camera placement for renderView1
renderView1.CameraPosition = [-22.133058935489576, -10.64508851958588, 64.50687193299979]
renderView1.CameraFocalPoint = [9.999999999999991, 9.999999999999998, 9.880000114440918]
renderView1.CameraViewUp = [0.19918727129422006, 0.8721825671628037, 0.44679077932704053]
renderView1.CameraParallelScale = 17.25150434777653

# save screenshot
SaveScreenshot(directoryOut+imageFilesOut+'.png', renderView1, ImageResolution=[1165, 804])

