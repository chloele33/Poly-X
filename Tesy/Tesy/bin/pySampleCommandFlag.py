# pySampleCommandFlag.py

import sys

import maya.api.OpenMaya as OpenMaya
# ... additional imports here ...
import maya.mel

import Tesy as sim
import numpy as np

import os
import json


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

kShortFlagJson = '-j'
kLongFlagJson = '-jsonFile'

kShortFlagJsonLoad = '-jl'
kLongFlagJsonLoad = '-jsonFileLoad'

kShortFlagReset = '-rd'
kLongFlagReset = '-resetData'

propLabel = 'none'


exampleScenes = []
exampleVecs = []

propagateScenes = []
propagateVecs = []


data = {}
data['group'] = []



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
            print("xminmaxyminmax", xmin, xmax, ymin, ymax)
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
                        groupData = {}
                        groupData["Group" + str(x+1)] = []
                        data['group'].append(groupData)
            	#scene = sim.Scene(group)
            	#poly = sim.Polygon(sim.vec2(xmin, ymin), sim.vec2(xmax, ymax), str(label))
                # prepare label information to group
                data['group'][int(group.split('p')[1])-1][group].append({"name": label,
                                                                         "xmin": xmin,
                                                                         "ymin": ymin,
                                                                         "xmax": xmax,
                                                                         "ymax": ymax})
                # data['group'][int(group.split('p')[1])-1][group].append({"xmin": xmin})
                # data['group'][int(group.split('p')[1]) - 1][group].append({"ymin": ymin})
                # data['group'][int(group.split('p')[1]) - 1][group].append({"xmax": xmax})
                # data['group'][int(group.split('p')[1]) - 1][group].append({"ymax": ymax})
                #maya.mel.eval( data['group'][0])
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
            print("ymin, ymax", xmin, xmax, ymin, ymax)
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

        if argData.isFlagSet(kShortFlagJson):
            filename = argData.flagArgumentString(kShortFlagJson, 0)

            with open(filename, "w+") as outfile:
                json.dump(data, outfile)

        if argData.isFlagSet(kShortFlagJsonLoad):
            # parse json file here
            filename = argData.flagArgumentString(kShortFlagJsonLoad, 0)

            with open (filename) as infile:
                indata = json.load(infile)
                numGroups = len(indata['group'])
                i = 0
                for gr in indata['group']:
                    for key in gr.keys():
                        groupName = key
                        groupNum = int(groupName[-1])
                        if (groupNum > 0):
                            if (groupNum > len(exampleScenes)):
                                for x in range(len(exampleScenes), groupNum):
                                    scene = sim.Scene("Group" + str(x))
                                    vec = sim.VecPoly()
                                    exampleScenes.append(scene)
                                    exampleVecs.append(vec)
                        for l in indata['group'][i][key]:
                            labelName = l['name']
                            xmin = l['xmin']
                            ymin = l['ymin']
                            xmax = l['xmax']
                            ymax = l['ymax']
                            exampleScenes[groupNum - 1].addPolygon(xmin, ymin, xmax, ymax, str(labelName),
                                                               exampleVecs[groupNum - 1])
                    i = i + 1

        if argData.isFlagSet(kShortFlagReset):
            del exampleScenes[:]
            del exampleVecs[:]
            del propagateScenes[:]
            del propagateVecs[:]
            data.clear()
            data['group'] = []

        if argData.isFlagSet(kShortFlagResult):
            finalResults = []
            for sIdx in range(len(propagateVecs)):
                vecPolyOut = propagateVecs[sIdx]
                output = propagateScenes[sIdx]
                allPossibleCandidates = []

                desiredLabel = exampleScenes[sIdx].desiredLabel
                # desiredpolygon in each scene
                desiredPolygons = []
                # GET CANDIDATE
                # For each example scene
                for vecPoly in exampleVecs:
                    print(vecPoly.size())
                    # iterate to find the target desired object
                    currPoly = None
                    for i in range(vecPoly.size()):
                        label = vecPoly[i].label
                        print(label)
                        if label == desiredLabel:
                            currPoly = vecPoly[i]
                            desiredPolygons.append(currPoly)
                    # print(currPoly.label)
                    # iterate through all other polygons to come up with relations
                    currminx = currPoly.low_bound.getX()
                    currmaxY = currPoly.low_bound.getY()
                    currmaxX = currPoly.upper_bound.getX()
                    currminy = currPoly.upper_bound.getY()
                    lowerBounds = []
                    upperBounds = []
                    for i in range(vecPoly.size()):
                        p = vecPoly[i]
                        label = p.label
                        if label != desiredLabel:
                            minx = p.low_bound.getX()
                            maxY = p.low_bound.getY()
                            maxX = p.upper_bound.getX()
                            miny = p.upper_bound.getY()
                            # print(minx, miny, maxX, maxY)
                            # print(currminx, currminy, currmaxX, currmaxY)
                            newMinX = minx - currminx
                            newMinY = miny - currminy
                            newMaxX = maxX - currmaxX
                            newMaxY = maxY - currmaxY
                            for j in range(vecPolyOut.size()):
                                if vecPolyOut[j].label == label:
                                    outputCurrPoly = vecPolyOut[j]
                                    outminx = outputCurrPoly.low_bound.getX()
                                    outmaxY = outputCurrPoly.low_bound.getY()
                                    outmaxX = outputCurrPoly.upper_bound.getX()
                                    outminy = outputCurrPoly.upper_bound.getY()
                                    lowerBounds.append([outminx - newMinX, outminy - newMinY])
                                    upperBounds.append([outmaxX - newMaxX, outmaxY - newMaxY])
                    print(lowerBounds)
                    print(upperBounds)
                    # Create candiate placements by taking combination of lower and upper
                    for lb in lowerBounds:
                        for ub in upperBounds:
                            allPossibleCandidates.append([lb[0], lb[1], ub[0], ub[1]])
                #
                # print(allPossibleCandidates)
                # print(len(allPossibleCandidates))
                # print(len(desiredPolygons))

                # get Polygons from candidate placements
                # get feature set x corresponding to each new candidate placement
                xList = []
                candidatePolygons = []
                for c in allPossibleCandidates:
                    print(c)
                    potentialPoly = sim.Polygon(sim.vec2(c[0], c[1]), sim.vec2(c[2], c[3]), desiredLabel)
                    candidatePolygons.append(potentialPoly)
                    xList.append(output.calculateRelationships(potentialPoly))

                print("here", xList)

                # get Phi values for each example
                phiList = []
                for i in range(len(exampleScenes)):
                    phiList.append(exampleScenes[i].calculateRelationships(desiredPolygons[i]))

                print("phi", phiList)

                # Get Similarity Measures
                similarityMeasures = sim.SimilarityMeasures()
                xphiList = []
                for x in xList:
                    x1 = []
                    for phi in phiList:
                        ss = similarityMeasures.shapeSimilarity(x, phi)
                        x1.append(ss)
                    xphiList.append(x1)

                print("xphiList", xphiList)

                # get Gram Matrix
                GramMatRows = []
                for i in phiList:
                    currRow = [];
                    for j in phiList:
                        currRow.append(similarityMeasures.shapeSimilarity(i, j))
                    GramMatRows.append(np.asarray(currRow))
                GramMat = np.asarray(GramMatRows)

                print("GRAM", GramMat)

                beta = 1.0
                I = np.eye(len(exampleScenes))
                covariance = GramMat + I / beta
                precision = np.linalg.inv(covariance)

                yx = np.matmul(xphiList, precision)
                print(yx)

                allRates = []
                for val in yx:
                    sum = 0
                    for v in val:
                        sum = sum + v
                    allRates.append(sum)

                maxVal = -np.inf
                maxValIndex = 0
                for i in range(len(allRates)):
                    if (allRates[i] > maxVal):
                        maxVal = allRates[i]
                        maxValIndex = i

                print(maxVal, maxValIndex)
                print(allRates)
                finalResults.append(allPossibleCandidates[maxValIndex])

            print(finalResults)
            for result in finalResults:
                results = result
                centerX = (results[0] + results[2]) / 2
                centerY = (results[1] + results[3]) / 2
                scaleX = results[2] - results[0]
                scaleY = abs(results[1] - results[3])
                maya.mel.eval("duplicate -un;")
                maya.mel.eval("scale -x " + str(scaleX) + " -z " + str(scaleY) + ";")
                maya.mel.eval("move -x " + str(centerX) + " -z " + str(centerY) +";")




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

    syntax.addFlag( kShortFlagJson, kLongFlagJson, OpenMaya.MSyntax.kString )
    syntax.addFlag(kShortFlagJsonLoad, kLongFlagJsonLoad, OpenMaya.MSyntax.kString)
    syntax.addFlag(kShortFlagReset, kLongFlagReset, OpenMaya.MSyntax.kDouble)

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