import os.path

class TrainingLog:
    def __init__(self, isVerbose, fileName = "training"):
        self._isVerbose = isVerbose
        self._fileName = fileName
        self._content = ""

    def Append(self, info):
        if type(info) != str or info == "":
            return
        self._content = self._content + info + "\n"
        if self._isVerbose:
            print(info)

    def Save(self, location):
        if self._content.strip() == "":
            return
        
        if type(location) != str or location.strip() == "" \
                or not os.path.isdir(location):
            return
        
        filePath = os.path.join(location, "{0}.log".format(self._fileName))
        with open(filePath, "w") as logFile:
            logFile.write(self._content)
