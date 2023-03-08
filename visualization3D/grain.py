# trace generated using paraview version 5.11.0
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
import sys, argparse



parser = argparse.ArgumentParser("paraview plot")
parser.add_argument("--sys", type=str, default = 'ls6')
parser.add_argument("--data_in", type=str, default = 'seed10000')
parser.add_argument("--data_dir", type=str, default = '/work/07428/ygqin/ls6/testGR/')
parser.add_argument("--save_name", type=str, default = '')

parser.add_argument('--clip', dest='clip', action='store_true')
parser.set_defaults(clip=False)
parser.add_argument('--cbar', dest='cbar', action='store_true')
parser.set_defaults(cbar=False)
parser.add_argument("--surface", type=int, default = 50)    
args = parser.parse_args()
    
if args.sys == 'ls6':
    directoryOut = '/scratch/07428/ygqin/graph/GrainGraphNN/visualization3D/'
if args.sys == 'mac':
    directoryOut = '/Users/yigongqin/Documents/Research/ML/Grain/GrainGraphNN/visualization3D/'
   # args.data_dir = directoryOut
    
datasetIn = args.data_dir + args.data_in + '.vtk' 
imageFilesOut = args.data_in + args.save_name
print("datasetIn = " + datasetIn)
print("directoryOut = " + directoryOut)
print("imageFilesOut = " + imageFilesOut)


#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'Legacy VTK Reader'
datavtk = LegacyVTKReader(registrationName='data.vtk', FileNames=[datasetIn])

# find settings proxy
colorPalette = GetSettingsProxy('ColorPalette')

# Properties modified on colorPalette
colorPalette.Text = [0.0, 0.0, 0.0]


if args.clip:
    
    datavtk = Clip(registrationName='datavtk', Input=datavtk)
    datavtk.ClipType = 'Plane'
    datavtk.HyperTreeGridClipper = 'Plane'
    datavtk.Scalars = ['POINTS', 'theta_z']
    datavtk.Value = 42.154137273875
    # Properties modified on datavtk.ClipType
    datavtk.ClipType.Origin = [10.0, 10.0, int(args.surface)]
    datavtk.ClipType.Normal = [0.0, 0.0, 1.0]



# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
datavtkDisplay = Show(datavtk, renderView1, 'UniformGridRepresentation')

# trace defaults for the display properties.
datavtkDisplay.Representation = 'Outline'
datavtkDisplay.ColorArrayName = ['POINTS', '']
datavtkDisplay.SelectTCoordArray = 'None'
datavtkDisplay.SelectNormalArray = 'None'
datavtkDisplay.SelectTangentArray = 'None'
datavtkDisplay.OSPRayScaleArray = 'theta_z'
datavtkDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
datavtkDisplay.SelectOrientationVectors = 'None'
datavtkDisplay.ScaleFactor = 2.0
datavtkDisplay.SelectScaleArray = 'theta_z'
datavtkDisplay.GlyphType = 'Arrow'
datavtkDisplay.GlyphTableIndexArray = 'theta_z'
datavtkDisplay.GaussianRadius = 0.1
datavtkDisplay.SetScaleArray = ['POINTS', 'theta_z']
datavtkDisplay.ScaleTransferFunction = 'PiecewiseFunction'
datavtkDisplay.OpacityArray = ['POINTS', 'theta_z']
datavtkDisplay.OpacityTransferFunction = 'PiecewiseFunction'
datavtkDisplay.DataAxesGrid = 'GridAxesRepresentation'
datavtkDisplay.PolarAxes = 'PolarAxesRepresentation'
datavtkDisplay.ScalarOpacityUnitDistance = 0.1385685404161374
datavtkDisplay.OpacityArrayName = ['POINTS', 'theta_z']
datavtkDisplay.IsosurfaceValues = [42.154137273875]
datavtkDisplay.SliceFunction = 'Plane'
datavtkDisplay.Slice = 123

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
datavtkDisplay.ScaleTransferFunction.Points = [0.75957012975, 0.0, 0.5, 0.0, 83.548704418, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
datavtkDisplay.OpacityTransferFunction.Points = [0.75957012975, 0.0, 0.5, 0.0, 83.548704418, 1.0, 0.5, 0.0]

# init the 'Plane' selected for 'SliceFunction'
datavtkDisplay.SliceFunction.Origin = [10.0, 10.0, 9.88]

# reset view to fit data
renderView1.ResetCamera(False)

# get the material library
materialLibrary1 = GetMaterialLibrary()

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(datavtkDisplay, ('POINTS', 'theta_z'))

# rescale color and/or opacity maps used to include current data range
datavtkDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
datavtkDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'theta_z'
theta_zLUT = GetColorTransferFunction('theta_z')

# get opacity transfer function/opacity map for 'theta_z'
theta_zPWF = GetOpacityTransferFunction('theta_z')

# change representation type
datavtkDisplay.SetRepresentationType('Surface')

# Properties modified on renderView1
renderView1.UseColorPaletteForBackground = 0

# Properties modified on renderView1
renderView1.Background = [1.0, 1.0, 1.0]


# get color legend/bar for theta_zLUT in view renderView1
theta_zLUTColorBar = GetScalarBar(theta_zLUT, renderView1)

# Properties modified on theta_zLUTColorBar
theta_zLUTColorBar.Title = '$\\theta_z$'
theta_zLUTColorBar.UseCustomLabels = 1
theta_zLUTColorBar.CustomLabels = [30.0, 60.0]
theta_zLUTColorBar.AddRangeLabels = 0
theta_zLUTColorBar.ScalarBarThickness = 32


if args.cbar:
    datavtkDisplay.SetScalarBarVisibility(renderView1, True)
else:
    datavtkDisplay.SetScalarBarVisibility(renderView1, False)
# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(1024, 1024)

# current camera placement for renderView1
renderView1.CameraPosition = [-22.133058935489576, -10.64508851958588, 64.50687193299979]
renderView1.CameraFocalPoint = [9.999999999999991, 9.999999999999998, 9.880000114440918]
renderView1.CameraViewUp = [0.19918727129422006, 0.8721825671628037, 0.44679077932704053]
renderView1.CameraParallelScale = 17.25150434777653

renderView1.ResetCamera()

# save screenshot
SaveScreenshot(directoryOut+imageFilesOut+'.pdf', renderView1, ImageResolution=[1024, 1024])

