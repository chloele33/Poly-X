#pragma once

#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "scene.h"

class SceneSet
{
public:

	std::map<std::string, Scene> scenes;

	SceneSet();


	void addPolyToScene(std::string label, Polygon poly);
};

