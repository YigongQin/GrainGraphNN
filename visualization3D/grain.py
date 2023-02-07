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
testvtk = LegacyVTKReader(registrationName='test.vtk', FileNames=[datasetIn])

# get active view

renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
testvtkDisplay = Show(testvtk, renderView1, 'UniformGridRepresentation')

# trace defaults for the display properties.
testvtkDisplay.Representation = 'Outline'
testvtkDisplay.ColorArrayName = ['POINTS', '']
testvtkDisplay.SelectTCoordArray = 'None'
testvtkDisplay.SelectNormalArray = 'None'
testvtkDisplay.SelectTangentArray = 'None'
testvtkDisplay.OSPRayScaleArray = 'theta_z'
testvtkDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
testvtkDisplay.SelectOrientationVectors = 'None'
testvtkDisplay.ScaleFactor = 2.0
testvtkDisplay.SelectScaleArray = 'theta_z'
testvtkDisplay.GlyphType = 'Arrow'
testvtkDisplay.GlyphTableIndexArray = 'theta_z'
testvtkDisplay.GaussianRadius = 0.1
testvtkDisplay.SetScaleArray = ['POINTS', 'theta_z']
testvtkDisplay.ScaleTransferFunction = 'PiecewiseFunction'
testvtkDisplay.OpacityArray = ['POINTS', 'theta_z']
testvtkDisplay.OpacityTransferFunction = 'PiecewiseFunction'
testvtkDisplay.DataAxesGrid = 'GridAxesRepresentation'
testvtkDisplay.PolarAxes = 'PolarAxesRepresentation'
testvtkDisplay.ScalarOpacityUnitDistance = 0.1385685404161374
testvtkDisplay.OpacityArrayName = ['POINTS', 'theta_z']
testvtkDisplay.ColorArray2Name = ['POINTS', 'theta_z']
testvtkDisplay.IsosurfaceValues = [42.154137273875]
testvtkDisplay.SliceFunction = 'Plane'
testvtkDisplay.Slice = 123
testvtkDisplay.SelectInputVectors = [None, '']
testvtkDisplay.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
testvtkDisplay.ScaleTransferFunction.Points = [0.75957012975, 0.0, 0.5, 0.0, 83.548704418, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
testvtkDisplay.OpacityTransferFunction.Points = [0.75957012975, 0.0, 0.5, 0.0, 83.548704418, 1.0, 0.5, 0.0]

# init the 'Plane' selected for 'SliceFunction'
testvtkDisplay.SliceFunction.Origin = [10.0, 10.0, 9.88]

# reset view to fit data
renderView1.ResetCamera(False)

# get the material library
materialLibrary1 = GetMaterialLibrary()

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(testvtkDisplay, ('POINTS', 'theta_z'))

# rescale color and/or opacity maps used to include current data range
testvtkDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
testvtkDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'theta_z'
theta_zLUT = GetColorTransferFunction('theta_z')

# get opacity transfer function/opacity map for 'theta_z'
theta_zPWF = GetOpacityTransferFunction('theta_z')

# get 2D transfer function for 'theta_z'
theta_zTF2D = GetTransferFunction2D('theta_z')

# change representation type
testvtkDisplay.SetRepresentationType('Surface')

# Properties modified on renderView1
renderView1.UseColorPaletteForBackground = 0

# Properties modified on renderView1
renderView1.Background = [1.0, 1.0, 1.0]

# Rescale transfer function
theta_zLUT.RescaleTransferFunction(0.0, 90.0)

# Rescale transfer function
theta_zPWF.RescaleTransferFunction(0.0, 90.0)

# Rescale 2D transfer function
theta_zTF2D.RescaleTransferFunction(0.0, 90.0, 0.0, 1.0)

# get color legend/bar for theta_zLUT in view renderView1
theta_zLUTColorBar = GetScalarBar(theta_zLUT, renderView1)

# Properties modified on theta_zLUTColorBar
theta_zLUTColorBar.UseCustomLabels = 1
theta_zLUTColorBar.CustomLabels = [0.0, 30.0, 60.0, 60.0]
theta_zLUTColorBar.AddRangeLabels = 0

# Properties modified on theta_zLUTColorBar
theta_zLUTColorBar.Title = '$\\theta_z$'
theta_zLUTColorBar.CustomLabels = [0.0, 30.0, 60.0, 90.0]

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

# get layout
layout1 = GetLayout()

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1626, 1378)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [-30.452731331138263, -10.370708981221844, 58.78254631264396]
renderView1.CameraFocalPoint = [10.000000000000002, 10.000000000000002, 9.880000114440918]
renderView1.CameraViewUp = [0.6395932442017959, 0.3601758751380124, 0.67911252450532]
renderView1.CameraParallelScale = 17.25150434777653

# save screenshot
SaveScreenshot(directoryOut+imageFilesOut+'.png', renderView1, ImageResolution=[1024, 1024])

"""
# layout/tab size in pixels
layout1.SetSize(1204, 689)

# current camera placement for renderView1
renderView1.CameraPosition = [2.154960958191303, 2.211460340678157, 2.834522189808974]
renderView1.CameraFocalPoint = [0.5, 0.5, 0.4959999918937683]
renderView1.CameraViewUp = [-0.4117211519433938, -0.5716378725636715, 0.7097294101932683]
renderView1.CameraParallelScale = 0.8637221728997225

# save animation
#SaveAnimation('/Users/yigongqin/Documents/Research/ML/Grain/visualization/animation.png', renderView1, ImageResolution=[1204, 689],
#    FrameWindow=[0, 9])

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1204, 689)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [2.154960958191303, 2.211460340678157, 2.834522189808974]
renderView1.CameraFocalPoint = [0.5, 0.5, 0.4959999918937683]
renderView1.CameraViewUp = [-0.4117211519433938, -0.5716378725636715, 0.7097294101932683]
renderView1.CameraParallelScale = 0.8637221728997225

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
"""