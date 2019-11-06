import matplotlib.pyplot as plt
import numpy as np
import os
from shutil import rmtree

class ChartsGenerator:
    def __init__(self, dataCollector, basePathForCharts):
        self._dataCollector = dataCollector
        self._basePathForCharts = basePathForCharts
        
        if os.path.isdir(self._basePathForCharts):
            self._pathToChartFiles = \
                    os.path.join(self._basePathForCharts, "charts")
            if os.path.isdir(self._pathToChartFiles):
                rmtree(self._pathToChartFiles)
            os.mkdir(self._pathToChartFiles)
        else:
            raise NotADirectoryError("ChartsGenerator.__init__() error: '{0}'" \
                    " does not exist!".format(self._basePathForCharts))
    
    def CreateAll(self):
        self.CreateBestComparisonCharts()
        self.CreateMeanComparisonCharts()
        self.CreateStdevComparisonCharts()
        self.CreateMeanTrainingTimeCharts()
        self.CreateValidationCharts()
        self.CreateMeanSearchCounterChart()
    
    def CreateBestComparisonCharts(self):
        lineLabels = ["DE", "PSO"]
        labelX = "episodes"
        labelY = "fitness"
        
        for trackNum in range(3):
            raceTrackName = "RaceTrack_{0}".format(trackNum + 1)
            dataFromTrack = self._dataCollector.BestFitness[raceTrackName]
            dataFromTrain_DE = dataFromTrack["DE"]
            dataFromTrain_PSO = dataFromTrack["PSO"]
            numOfTrials = len(dataFromTrain_DE)
            
            for trialNum in range(numOfTrials):
                fileName = os.path.join(
                        self._pathToChartFiles,
                        "best_track_{0}_trial_{1}.svg".format(
                            trackNum + 1,
                            str(trialNum + 1).zfill(2)))
                chartData = \
                        [dataFromTrain_DE[trialNum], dataFromTrain_PSO[trialNum]]
                chartTitle = "Best fitness values - Track {0}, Trial {1}" \
                        .format(trackNum + 1, trialNum + 1)
                
                self._drawLineChartWithMarkers(
                        fileName = fileName,
                        chartData = chartData,
                        lineLabels = lineLabels,
                        chartTitle = chartTitle,
                        labelX = labelX,
                        labelY = labelY)
    
    def CreateMeanComparisonCharts(self):
        lineLabels = ["DE", "PSO_Pbest", "PSO_Episode"]
        labelX = "episodes"
        labelY = "mean fitness"
        
        for trackNum in range(3):
            raceTrackName = "RaceTrack_{0}".format(trackNum + 1)
            dataFromTrack = self._dataCollector.MeanFitness[raceTrackName]
            dataFromTrain_DE = dataFromTrack["DE"]
            dataFromTrain_PSO_Pest = dataFromTrack["PSO_Pbest"]
            dataFromTrain_PSO_Episode = dataFromTrack["PSO_Episode"]
            numOfTrials = len(dataFromTrain_DE)
            
            for trialNum in range(numOfTrials):
                fileName = os.path.join(
                        self._pathToChartFiles,
                        "mean_track_{0}_trial_{1}.svg".format(
                            trackNum + 1,
                            str(trialNum + 1).zfill(2)))
                chartData = [
                        dataFromTrain_DE[trialNum],
                        dataFromTrain_PSO_Pest[trialNum],
                        dataFromTrain_PSO_Episode[trialNum]]
                chartTitle = "Mean fitness values - Track {0}, Trial {1}" \
                        .format(trackNum + 1, trialNum + 1)
                
                self._drawLineChartWithMarkers(
                        fileName = fileName,
                        chartData = chartData,
                        lineLabels = lineLabels,
                        chartTitle = chartTitle,
                        labelX = labelX,
                        labelY = labelY)
    
    def CreateStdevComparisonCharts(self):
        lineLabels = ["DE", "PSO_Pbest", "PSO_Episode"]
        labelX = "episodes"
        labelY = "stdev of fitness"
        
        for trackNum in range(3):
            raceTrackName = "RaceTrack_{0}".format(trackNum + 1)
            dataFromTrack = self._dataCollector.StdevFitness[raceTrackName]
            dataFromTrain_DE = dataFromTrack["DE"]
            dataFromTrain_PSO_Pest = dataFromTrack["PSO_Pbest"]
            dataFromTrain_PSO_Episode = dataFromTrack["PSO_Episode"]
            numOfTrials = len(dataFromTrain_DE)
            
            for trialNum in range(numOfTrials):
                fileName = os.path.join(
                        self._pathToChartFiles,
                        "stdev_track_{0}_trial_{1}.svg".format(
                            trackNum + 1,
                            str(trialNum + 1).zfill(2)))
                chartData = [
                        dataFromTrain_DE[trialNum],
                        dataFromTrain_PSO_Pest[trialNum],
                        dataFromTrain_PSO_Episode[trialNum]]
                chartTitle = "Standard Deviation - Track {0}, Trial {1}" \
                        .format(trackNum + 1, trialNum + 1)
                
                self._drawLineChartWithMarkers(
                        fileName = fileName,
                        chartData = chartData,
                        lineLabels = lineLabels,
                        chartTitle = chartTitle,
                        labelX = labelX,
                        labelY = labelY)
    
    def CreateMeanTrainingTimeCharts(self):
        barLabels = ["DE", "PSO"]
        tickLabelsOnX = ["RaceTrack_1", "RaceTrack_2", "RaceTrack_3"]
        
        labelY_TimeInSeconds = "Time (s)"
        fileName_TimeInSeconds = \
                os.path.join(self._pathToChartFiles, "train_time_seconds.svg")
        chartData_TimeInSeconds = self._calculateMeanValuesForData(
                self._dataCollector.MeanTrainingTimesAsSeconds)
        chartTitle_TimeInSeconds = "Mean time of training in seconds"
        
        self._drawGroupedBarChart(
                fileName = fileName_TimeInSeconds,
                chartData = chartData_TimeInSeconds,
                barLabels = barLabels,
                tickLabelsOnX = tickLabelsOnX,
                chartTitle = chartTitle_TimeInSeconds,
                labelY = labelY_TimeInSeconds,
                doesPrintBarValues = True)
        
        labelY_TimeInEpisodes = "Time (episodes)"
        fileName_TimeInEpisodes = \
                os.path.join(self._pathToChartFiles, "train_time_episodes.svg")
        chartData_TimeInEpisodes = self._calculateMeanValuesForData(
                self._dataCollector.MeanTrainingTimesAsEpisodes)
        chartTitle_TimeInEpisodes = "Mean time of training in episodes"
        
        self._drawGroupedBarChart(
                fileName = fileName_TimeInEpisodes,
                chartData = chartData_TimeInEpisodes,
                barLabels = barLabels,
                tickLabelsOnX = tickLabelsOnX,
                chartTitle = chartTitle_TimeInEpisodes,
                labelY = labelY_TimeInEpisodes,
                doesPrintBarValues = True,
                floatPrecision = 0)
        
    def CreateValidationCharts(self):
        barLabels = ["Run_1", "Run_2", "Run_3"]
        tickLabelsOnX = ["Train_1", "Train_2", "Train_3"]
        labelY = "Success Runs"
        
        fileName_DE = os.path.join(self._pathToChartFiles, "validation_de.svg")
        chartData_DE = self._dataCollector.ValidationMatrices["DE"]
        chartTitle_DE = "Validation for Differential Evolution"
        
        self._drawGroupedBarChart(
                fileName = fileName_DE,
                chartData = chartData_DE,
                barLabels = barLabels,
                tickLabelsOnX = tickLabelsOnX,
                chartTitle = chartTitle_DE,
                labelY = labelY,
                doesPrintBarValues = True,
                floatPrecision = 0)
        
        fileName_PSO = os.path.join(self._pathToChartFiles, "validation_pso.svg")
        chartData_PSO = self._dataCollector.ValidationMatrices["PSO"]
        chartTitle_PSO = "Validation for Particle Swarm Optimization"
        
        self._drawGroupedBarChart(
                fileName = fileName_PSO,
                chartData = chartData_PSO,
                barLabels = barLabels,
                tickLabelsOnX = tickLabelsOnX,
                chartTitle = chartTitle_PSO,
                labelY = labelY,
                doesPrintBarValues = True,
                floatPrecision = 0)
    
    def CreateMeanSearchCounterChart(self):
        barLabels = ["DE", "PSO"]
        tickLabelsOnX = ["RaceTrack_1", "RaceTrack_2", "RaceTrack_3"]
        labelY = "Candidate Solutions"
        
        fileName = os.path.join(self._pathToChartFiles, "search_count.svg")
        chartData = self._calculateMeanValuesForData(
                self._dataCollector.MeanSearchCounters)
        chartTitle = "Mean size of search pool before solution was found"
        
        self._drawGroupedBarChart(
                fileName = fileName,
                chartData = chartData,
                barLabels = barLabels,
                tickLabelsOnX = tickLabelsOnX,
                chartTitle = chartTitle,
                labelY = labelY,
                doesPrintBarValues = True,
                floatPrecision = 0)

    def _calculateMeanValuesForData(self, data):
        resultData = [[0 for _ in range(2)] for _ in range(3)]
        
        for i in range(3):
            keyName = "RaceTrack_{0}".format(i+1)
            resultData[i][0] = \
                    data[keyName]["DE"]["Sum"] / data[keyName]["DE"]["TrialCounter"]
            resultData[i][1] = \
                    data[keyName]["PSO"]["Sum"] / data[keyName]["PSO"]["TrialCounter"]

        return resultData

    def _drawLineChartWithMarkers(
            self,
            fileName,   # string
            chartData,  # 2D list (lineClass, episode)
            lineLabels,    # list of strings
            chartTitle = "",    # string
            labelX = "",          # string
            labelY = ""):          # string
        numOfLineClasses = len(chartData)
        labelStrings = self._prepareLabelStrings(lineLabels, chartData)
        
        figureSize = (10, 8)
        figure, axes = plt.subplots(figsize = figureSize)
        
        axes.tick_params(axis='x', which='both', labelsize = 7)
        
        for i in range(numOfLineClasses):
            axes.plot(
                    chartData[i],
                    's--',
                    linewidth = 0.8,
                    markersize = 5,
                    label = labelStrings[i])
        
        axes.set_title(chartTitle, fontsize = 'x-large')
        axes.set_xlabel(labelX, fontsize = 'large')
        axes.set_ylabel(labelY, fontsize = 'large')
        
        start, end = axes.get_xlim()
        axes.xaxis.set_ticks(np.arange(0, end, 1))
        
        start, end = axes.get_ylim()
        axes.yaxis.set_ticks(np.arange(0, end, 5))
        
        axes.legend(
                loc = "upper left",
                bbox_to_anchor=(0, -0.2),
                fontsize = 'large',
                markerscale = 1.5)
        figure.tight_layout(pad = 0.4)
        figure.savefig(fileName, format = "svg", transparent = True)
        plt.close('all')

    def _prepareLabelStrings(self, lineLabels, chartData):
        labelStrings = []
        for label, lineData in zip(lineLabels, chartData):
            tempString = "{0}: ".format(label)
            markerCounter = 1
            for markerValue in lineData:
                if (markerCounter % 15) == 0:
                    tempString += "\n"
                tempString += "{0:.2f} - ".format(markerValue)
                markerCounter += 1
            
            tempString = tempString.strip(' -')
            labelStrings.append(tempString)
        
        return labelStrings

    def _drawGroupedBarChart(
            self,
            fileName,   # string
            chartData,  # 2D list (barClass, tickX)
            barLabels,    # list of strings
            tickLabelsOnX,   # list of strings
            chartTitle = "",    # string
            labelX = "",          # string
            labelY = "",          # string
            doesPrintBarValues = False,
            floatPrecision = 2): # boolean
        ticksOnX = np.arange(len(tickLabelsOnX))
        numOfBarClasses = len(chartData[0])
        
        if numOfBarClasses > 1:
            barsWidth = 0.4 / (numOfBarClasses - 1)
        else:
            barsWidth = 0.4
        
        figure, axes = plt.subplots()
        barClasses = []
        for i in range(numOfBarClasses):
            tempBarClass = axes.bar(
                    ticksOnX + (i*2 - (numOfBarClasses*2 - 1 - numOfBarClasses)) * barsWidth / 2,
                    [round(tickData[i], floatPrecision) for tickData in chartData],
                    barsWidth,
                    label = barLabels[i])
            barClasses.append(tempBarClass)
        
        axes.set_title(chartTitle)
        axes.set_xlabel(labelX)
        axes.set_ylabel(labelY)
        axes.set_xticks(ticksOnX)
        axes.set_xticklabels(tickLabelsOnX)
        axes.legend(loc = "lower center", bbox_to_anchor=(1.15, 0.1))
        
        if doesPrintBarValues:
            for barClass in barClasses:
                self._displayBarsValues(barClass, axes, floatPrecision)
        
        figure.tight_layout()
        figure.savefig(
                fileName,
                format = "svg",
                transparent = True,
                bbox_inches='tight')
        plt.close('all')
        
    def _displayBarsValues(self, bars, axes, floatPrecision):
        for bar in bars:
            height = bar.get_height()
            axes.annotate(
                    '{0:.{1}f}'.format(height, floatPrecision),
                    xy = (bar.get_x() + bar.get_width() / 2, height),
                    xytext = (0, 3),
                    textcoords = "offset points",
                    ha = 'center',
                    va = 'bottom')
