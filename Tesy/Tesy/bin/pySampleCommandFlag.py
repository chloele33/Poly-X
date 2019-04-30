# pySampleCommandFlag.py

import sys

import maya.api.OpenMaya as OpenMaya
# ... additional imports here ...
import maya.mel

import Tesy as sim



kPluginCmdName = 'polyX'

kShortFlagGroup = '-g'
kLongFlagGroup = '-group'

kShortFlagGroupProp = '-gp'
kLongFlagGroupProp = '-groupProp'

kShortFlagLabel = '-l'
kLongFlagLabel = '-label'

kShortFlagXMin = '-xm'
kLongFlagXMin = '-xMin'

kShortFlagXMax = '-xmx'
kLongFlagXMax = '-xMax'

kShortFlagYMin = '-ym'
kLongFlagYMin = '-yMin'

kShortFlagYMax = '-ymx'
kLongFlagYMax = '-yMax'


kShortFlagPropLabel = '-p'
kLongFlagPropLabel = '-propLabel'

kShortFlagResult = '-r'
kLongFlagResult = '-result'

#Prop_label = 'mirror'


exampleScenes = []
exampleVecs = []

propagateScenes = []
propagateVecs = []



def maya_useNewAPI():
	"""
	The presence of this function tells Maya that the plugin produces, and
	expects to be passed, objects created using the Maya Python API 2.0.
	"""
	pass
	
##########################################################
# Plug-in 
##########################################################
class MyCommandWithFlagClass( OpenMaya.MPxCommand ):
    
    def __init__(self):
        ''' Constructor. '''
        OpenMaya.MPxCommand.__init__(self)
    
    def doIt(self, args):
        ''' Command execution. '''
        
        # We recommend parsing your arguments first.
        self.parseArguments( args )

        # Remove the following 'pass' keyword and replace it with the code you want to run. 
        pass
    
    def parseArguments(self, args):
        ''' 
        The presence of this function is not enforced,
        but helps separate argument parsing code from other
        command code. 
        '''
        
        # The following MArgParser object allows you to check if specific flags are set.
        argData = OpenMaya.MArgParser( self.syntax(), args )
        
        # Get the information for the example scenes
        if argData.isFlagSet( kShortFlagGroup ) and argData.isFlagSet(kShortFlagLabel) and argData.isFlagSet(kShortFlagXMin) \
            and argData.isFlagSet(kShortFlagXMax) and argData.isFlagSet(kShortFlagYMin) and argData.isFlagSet(kShortFlagYMax):
            maya.mel.eval("print \"It Works!\"")
            group = argData.flagArgumentString(kShortFlagGroup, 0)
            label = argData.flagArgumentString(kShortFlagLabel, 0)
            xmin = argData.flagArgumentFloat(kShortFlagXMin, 0)
            xmax = argData.flagArgumentFloat(kShortFlagXMax, 0)
            ymin = argData.flagArgumentFloat(kShortFlagYMin, 0)
            ymax = argData.flagArgumentFloat(kShortFlagYMax, 0)
            maya.mel.eval("print \"Group: " + group + "\"")
            maya.mel.eval("print \"Label: " + label + "\"")
            maya.mel.eval("print \"Xmin: " + str(xmin) + "\"")
            # In this case, we print the passed flag's value as an integer.
            # We use the '0' to index the flag's first and only parameter.
            #flagValue = argData.flagArgumentInt( kShortFlagName, 0 )
            #print kLongFlagName + ': ' + str( flagValue )
            groupNum = int(group[-1])
            if (groupNum > 0):
            	if (groupNum > len(exampleScenes)):
            		for x in range(len(exampleScenes), groupNum):
            			scene = sim.Scene("Group" + str(x))
            			vec = sim.VecPoly()
            			exampleScenes.append(scene)
            			exampleVecs.append(vec)
            	#scene = sim.Scene(group)
            	#poly = sim.Polygon(sim.vec2(xmin, ymin), sim.vec2(xmax, ymax), str(label))
            	exampleScenes[groupNum - 1].addPolygon(xmin, ymin, xmax, ymax, str(label), exampleVecs[groupNum - 1])


        # Get the information for the scenes we will be propagating
        if argData.isFlagSet( kShortFlagGroupProp ) and argData.isFlagSet(kShortFlagLabel) and argData.isFlagSet(kShortFlagXMin) \
            and argData.isFlagSet(kShortFlagXMax) and argData.isFlagSet(kShortFlagYMin) and argData.isFlagSet(kShortFlagYMax):
            maya.mel.eval("print \"It Works!\"")
            group = argData.flagArgumentString(kShortFlagGroupProp, 0)
            label = argData.flagArgumentString(kShortFlagLabel, 0)
            xmin = argData.flagArgumentFloat(kShortFlagXMin, 0)
            xmax = argData.flagArgumentFloat(kShortFlagXMax, 0)
            ymin = argData.flagArgumentFloat(kShortFlagYMin, 0)
            ymax = argData.flagArgumentFloat(kShortFlagYMax, 0)
            maya.mel.eval("print \"Group: " + group + "\"")
            maya.mel.eval("print \"Label: " + label + "\"")
            maya.mel.eval("print \"Xmin: " + str(xmin) + "\"")
            # In this case, we print the passed flag's value as an integer.
            # We use the '0' to index the flag's first and only parameter.
            #flagValue = argData.flagArgumentInt( kShortFlagName, 0 )
            #print kLongFlagName + ': ' + str( flagValue )
            groupNum = int(group[-1])
            if (groupNum > 0):
            	if (groupNum > len(propagateScenes)):
            		for x in range(len(propagateScenes), groupNum):
            			scene = sim.Scene("Group" + str(x))
            			vec = sim.VecPoly()
            			propagateScenes.append(scene)
            			propagateVecs.append(vec)
            	#scene = sim.Scene(group)
            	#poly = sim.Polygon(sim.vec2(xmin, ymin), sim.vec2(xmax, ymax), str(label))
            	propagateScenes[groupNum - 1].addPolygon(xmin, ymin, xmax, ymax, str(label), propagateVecs[groupNum - 1])

        if argData.isFlagSet(kShortFlagPropLabel):
        	propLabel = argData.flagArgumentString(kShortFlagPropLabel, 0)
        	for i in range(0, len(exampleScenes) - 1):
        		exampleScenes[i].desiredLabel = str(propLabel)
        	maya.mel.eval("print \"It Works!\"")

        if argData.isFlagSet(kShortFlagResult):
        	results = [2.0, 1.0, 4.5, 6.2]
        	centerX = (results[0] + results[2]) / 2
        	centerY = (results[1] + results[3]) / 2
        	scaleX = results[2] - centerX
        	scaleY = results[3] - centerY
        	maya.mel.eval("duplicate -un;")
        	maya.mel.eval("move -x " + str(centerX) + " -y " + str(centerY) +";")
        	maya.mel.eval("scale -x " + str(scaleX) + " -y " + str(scaleY) + ";")
            
            
        
        # ... If there are more flags, process them here ...

##########################################################
# Plug-in initialization.
##########################################################
def cmdCreator():
    ''' Create an instance of our command. '''
    return MyCommandWithFlagClass() 

def syntaxCreator():
    ''' Defines the argument and flag syntax for this command. '''
    syntax = OpenMaya.MSyntax()
    
    # In this example, our flag will be expecting a numeric value, denoted by OpenMaya.MSyntax.kDouble. 
    syntax.addFlag( kShortFlagGroup, kLongFlagGroup, OpenMaya.MSyntax.kString )
    syntax.addFlag( kShortFlagLabel, kLongFlagLabel, OpenMaya.MSyntax.kString )
    syntax.addFlag( kShortFlagXMin, kLongFlagXMin, OpenMaya.MSyntax.kDouble )
    syntax.addFlag( kShortFlagXMax, kLongFlagXMax, OpenMaya.MSyntax.kDouble )
    syntax.addFlag( kShortFlagYMin, kLongFlagYMin, OpenMaya.MSyntax.kDouble )
    syntax.addFlag( kShortFlagYMax, kLongFlagYMax, OpenMaya.MSyntax.kDouble )

    syntax.addFlag( kShortFlagPropLabel, kLongFlagPropLabel, OpenMaya.MSyntax.kString )

    syntax.addFlag( kShortFlagGroupProp, kLongFlagGroupProp, OpenMaya.MSyntax.kString )

    syntax.addFlag( kShortFlagResult, kLongFlagResult, OpenMaya.MSyntax.kDouble )
    
    # ... Add more flags here ...
        
    return syntax
    
def initializePlugin( mobject ):
    ''' Initialize the plug-in when Maya loads it. '''
    mplugin = OpenMaya.MFnPlugin( mobject )
    try:
        mplugin.registerCommand( kPluginCmdName, cmdCreator, syntaxCreator )
    except:
        sys.stderr.write( 'Failed to register command: ' + kPluginCmdName )

    maya.mel.eval("source \"" + mplugin.loadPath() + "/menu.mel\"")


def uninitializePlugin( mobject ):
    ''' Uninitialize the plug-in when Maya un-loads it. '''
    mplugin = OpenMaya.MFnPlugin( mobject )
    try:
        mplugin.deregisterCommand( kPluginCmdName )
    except:
        sys.stderr.write( 'Failed to unregister command: ' + kPluginCmdName )

    maya.mel.eval("deleteUI $menu;");

##########################################################
# Sample usage.
##########################################################
''' 
# Copy the following lines and run them in Maya's Python Script Editor:

import maya.cmds as cmds
cmds.loadPlugin( 'sampleCommandFlag.py' )
cmds.myCommandWithFlag( myFlag = 4 )

'''