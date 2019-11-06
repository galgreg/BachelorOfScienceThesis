using UnityEditor;
using System;

class EnvBuilder {
    public static void Build() {
        var buildParameters = Environment.GetCommandLineArgs();
        var numOfParameters = buildParameters.Length;
        var baseLocation = buildParameters[numOfParameters - 2];
        var target = buildParameters[numOfParameters - 1];
        var playerOptionSet = GetPlayerOptionSet(baseLocation, target);
        foreach (var playerOptions in playerOptionSet) {
            BuildPipeline.BuildPlayer(playerOptions);
        }
    }

    private static BuildPlayerOptions[] GetPlayerOptionSet(
            string aBaseLocation,
            string aTargetName) {
        var scenes = GetScenes();
        var buildTarget = GetBuildTarget(aTargetName);
        var fileExtension = GetFileExtension(buildTarget);
        BuildPlayerOptions[] playerOptions = new BuildPlayerOptions[3];
        

        for (uint i = 0; i < 3; ++i) {
            BuildPlayerOptions tempOptions = new BuildPlayerOptions();
            tempOptions.scenes = new string[]{ scenes[i] };

            string locationToBuild =
                    string.Format(
                            aBaseLocation + "/RaceTrack_{0}/RaceTrack_{0}.{1}",
                            i+1,
                            fileExtension);
            tempOptions.locationPathName = locationToBuild;
            
            tempOptions.target = buildTarget;

            BuildOptions buildOptions = BuildOptions.EnableHeadlessMode;
            tempOptions.options = buildOptions; 

            playerOptions[i] = tempOptions;
        }
        return playerOptions;
    }

    private static string[] GetScenes() {
        string[] scenes = {
                "Assets/Scenes/RaceTrack_1.unity",
                "Assets/Scenes/RaceTrack_2.unity",
                "Assets/Scenes/RaceTrack_3.unity" };
        return scenes;
    }

    private static BuildTarget GetBuildTarget(string aTargetName) {
        BuildTarget buildTarget = new BuildTarget();
        if (aTargetName == "Linux") {
            buildTarget = BuildTarget.StandaloneLinux64;
        } else if (aTargetName == "Windows") {
            buildTarget = BuildTarget.StandaloneWindows64;
        } else {
            throw new ArgumentException("EnvBuilder.GetBuildOptions() error: "
                    + "Wrong target value!");
        }
        return buildTarget;
    }

    private static string GetFileExtension(BuildTarget buildTarget) {
        if (buildTarget == BuildTarget.StandaloneLinux64) {
            return "x86_64";
        } else if (buildTarget == BuildTarget.StandaloneWindows64) {
            return "exe";
        } else {
            throw new ArgumentException("EnvBuilder.GetFileExtension() error: "
                    + "Wrong buildTarget!");
        }
    }
}