import sys

# Imports to use the Maya Python API
import maya.OpenMaya as OpenMaya
import maya.OpenMayaMPx as OpenMayaMPx

# Import the Python wrappers for MEL commands
import maya.cmds as cmds

import maya.mel
import os

import Tesy

#import test as ui

# The name of the command.
kPluginCmdName = "pyPolyX"

exampleObjects = []


def addExampleObject(group, label, objectName):
	groupNum = int(group[-1])
	if (groupNum > 0):
		if (groupNum > len(exampleObjects)):
			for x in range(len(exampleObjects), groupNum):
				exampleObjects.append({})
		exampleObjects[groupNum - 1][label] = objectName


#def mainWindow():
#	print("mainWindow")

#def helpWindow():
#	print("helpWindow")

#def aboutWindow():
#	print("aboutwindow")

#def createUI():

#	menuId = "polyMenu"
#	cmds.menu(menuId, label='Poly-X', parent='MayaWindow')
#	cmds.menuItem(label='Poly-X Window', command='mainWindow()')
#	cmds.menuItem(label='Help', command='helpWindow()')
#	cmds.menuItem(label='About', command='aboutWindow()')


class polyXCommand(OpenMayaMPx.MPxCommand):
    def __init__(self):
        OpenMayaMPx.MPxCommand.__init__(self)

    def doIt(self, argList):
        # TODO 

        self.setResult("Executed command")

# Create an instance of the command.
def cmdCreator():
    return OpenMayaMPx.asMPxPtr(helloMayaCommand())

# Initialize the plugin
def initializePlugin(mobject):
    mplugin = OpenMayaMPx.MFnPlugin(mobject, "polyX", "1.0", "2019")
    try:
        mplugin.registerCommand(kPluginCmdName, cmdCreator)
    except:
        sys.stderr.write("Failed to register command: %s\n" % kPluginCmdName)
        raise

    #addExampleObject("Group0", "test", "test")

	#maya.mel.eval("global proc okLabel( ) { " + "$group = `optionMenu -q -v groupMenu`; " + "$name = \"scroll\" + $group; " + 
	#	"$label = `textField -q -text objectLabel`; " + "$parent = \"layout\" + $group; " + "$shapeName = `ls -sl -o`; " + 
	#	"if (size($shapeName)) { " + "python(\"addExampleObject(\"Group2\", \"sink\", \"square3\")\"); " + "if (!`textScrollList -exists $name`) { " + 
	#	"textScrollList -append $label -parent $parent $name; " + "for ($i = 0; $i < size($shapeName); $i++) { " + 
	#	"python(\"addExampleObject(\" + $group + \", \" +  $label + \", \" + $shapeName[$i] + \")\"); } } " + "else { textScrollList -e -append $label $name; } " + 
	#	"deleteUI labelWindow; } " + "else { } };")
	
	#maya.mel.eval("$group = `optionMenu -q -v groupMenu`;");
	#maya.mel.eval("$name = \"scroll\" + $group;");
	#maya.mel.eval("$label = `textField -q -text objectLabel`;");
	#maya.mel.eval("$parent = \"layout\" + $group;");

	#maya.mel.eval("$shapeName = `ls -sl -o`;");
	#maya.mel.eval("if (size($shapeName)) {");
	#maya.mel.eval("python(\"addExampleObject(\"Group2\", \"sink\", \"square3\")\");");
	#maya.mel.eval("if (!`textScrollList -exists $name`) {");
	#maya.mel.eval("textScrollList -append $label -parent $parent $name;");
	#maya.mel.eval("for ($i = 0; $i < size($shapeName); $i++) { python(\"addExampleObject(\" + $group + \", \" +  $label + \", \" + $shapeName[$i] + \")\"); }");
	#maya.mel.eval("}");
	#maya.mel.eval("else { textScrollList -e -append $label $name; }");

	#maya.mel.eval("deleteUI labelWindow;");
	#maya.mel.eval("}");
	#maya.mel.eval("else { }");
	#maya.mel.eval("};");


	#maya.mel.eval("global proc okLabel( ) { }")
	#maya.mel.eval("window -title \"Label Object\" -p MayaWindow labelWindow; " + "frameLayout -labelVisible false -marginWidth 200; " + 
	#	"columnLayout; " + "textField -w 200 -pht \"Label selected object\" objectLabel;" + "button -label \"Ok\" -w 100 -command \"okLabel()\"; " + 
	#	"setParent ..; " + "setParent ..; " + "showWindow labelWindow;")
    maya.mel.eval("source \"" + mplugin.loadPath() + "/menu.mel\"")
    #ui.createUI()
    #print(exampleObjects)

# Uninitialize the plugin
def uninitializePlugin(mobject):
    mplugin = OpenMayaMPx.MFnPlugin(mobject)
    try:
        mplugin.deregisterCommand(kPluginCmdName)
    except:
        sys.stderr.write("Failed to unregister command: %s\n" % kPluginCmdName)
        raise

    maya.mel.eval("deleteUI $menu;");
    #menuId = "polyMenu"
    #cmds.deleteUI(menuId)
    #ui.deleteMenu()
