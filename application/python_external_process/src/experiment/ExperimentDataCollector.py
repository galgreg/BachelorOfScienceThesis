class ExperimentDataCollector:
    def __init__(self):
        self.BestFitness = {
            "RaceTrack_1" : {
                "DE": [],
                "PSO" : []
            },
            "RaceTrack_2" : {
                "DE": [],
                "PSO" : []
            },
            "RaceTrack_3" : {
                "DE": [],
                "PSO" : []
            }
        }
        self.MeanFitness = {
            "RaceTrack_1" : {
                "DE": [],
                "PSO_Pbest" : [],
                "PSO_Episode" : [],
            },
            "RaceTrack_2" : {
                "DE": [],
                "PSO_Pbest" : [],
                "PSO_Episode" : [],
            },
            "RaceTrack_3" : {
                "DE": [],
                "PSO_Pbest" : [],
                "PSO_Episode" : [],
            }
        }
        self.StdevFitness = {
            "RaceTrack_1" : {
                "DE": [],
                "PSO_Pbest" : [],
                "PSO_Episode" : [],
            },
            "RaceTrack_2" : {
                "DE": [],
                "PSO_Pbest" : [],
                "PSO_Episode" : [],
            },
            "RaceTrack_3" : {
                "DE": [],
                "PSO_Pbest" : [],
                "PSO_Episode" : [],
            }
        }
        self.MeanTrainingTimesAsSeconds = {
            "RaceTrack_1" : {
                "DE": {
                    "Time" : 0.0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "Time" : 0.0,
                    "TrialCounter" : 0
                }
            },
            "RaceTrack_2" : {
                "DE": {
                    "Time" : 0.0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "Time" : 0.0,
                    "TrialCounter" : 0
                }
            },
            "RaceTrack_3" : {
                "DE": {
                    "Time" : 0.0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "Time" : 0.0,
                    "TrialCounter" : 0
                }
            }
        }
        self.MeanTrainingTimesAsEpisodes = {
            "RaceTrack_1" : {
                "DE": {
                    "Episodes" : 0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "Episodes" : 0,
                    "TrialCounter" : 0
                }
            },
            "RaceTrack_2" : {
                "DE": {
                    "Episodes" : 0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "Episodes" : 0,
                    "TrialCounter" : 0
                }
            },
            "RaceTrack_3" : {
                "DE": {
                    "Episodes" : 0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "Episodes" : 0,
                    "TrialCounter" : 0
                }
            }
        }
        self.ValidationMatrices = {
            "PSO" : [[0] * 3] * 3,
            "DE" : [[0] * 3] * 3
        }
        self.MeanSearchCounters = {
            "RaceTrack_1" : {
                "DE": {
                    "SearchCounter" : 0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "SearchCounter" : 0,
                    "TrialCounter" : 0
                }
            },
            "RaceTrack_2" : {
                "DE": {
                    "SearchCounter" : 0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "SearchCounter" : 0,
                    "TrialCounter" : 0
                }
            },
            "RaceTrack_3" : {
                "DE": {
                    "SearchCounter" : 0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "SearchCounter" : 0,
                    "TrialCounter" : 0
                }
            }
        }

    def AppendBestFitnessSequence(self, trackNum, algorithm, fitnessSequence):
        raceTrackName = "RaceTrack_{0}".format(trackNum)
        self.BestFitness[raceTrackName][algorithm].append(fitnessSequence)
    
    def AppendMeanFitnessSequence(self, trackNum, algorithm, meanSequence):
        raceTrackName = "RaceTrack_{0}".format(trackNum)
        self.MeanFitness[raceTrackName][algorithm].append(meanSequence)
    
    def AppendStdevFitnessSequence(self, trackNum, algorithm, stdevSequence):
        raceTrackName = "RaceTrack_{0}".format(trackNum)
        self.StdevFitness[raceTrackName][algorithm].append(stdevSequence)
    
    def AddTimeInSecondsFromTraining(self, trackNum, algorithm, time):
        raceTrackName = "RaceTrack_{0}".format(trackNum)
        self.MeanTrainingTimesAsSeconds[raceTrackName][algorithm]["Time"] += time
        self.MeanTrainingTimesAsSeconds[raceTrackName][algorithm]["TrialCounter"] += 1
    
    def AddTimeInEpisodesFromTraining(self, trackNum, algorithm, time):
        raceTrackName = "RaceTrack_{0}".format(trackNum)
        self.MeanTrainingTimesAsEpisodes[raceTrackName][algorithm]["Episodes"] += time
        self.MeanTrainingTimesAsEpisodes[raceTrackName][algorithm]["TrialCounter"] += 1

    def IncrementValidationMatrixEntry(self, algorithm, trainNum, runNum):
        self.ValidationMatrices[algorithm][trainNum-1][runNum-1] += 1

    def AddToSearchCounter(self, trackNum, algorithm, numOfSearches):
        raceTrackName = "RaceTrack_{0}".format(trackNum)
        self.MeanSearchCounters[raceTrackName][algorithm]["SearchCounter"] += numOfSearches
        self.MeanSearchCounters[raceTrackName][algorithm]["TrialCounter"] += 1
