import re

class Settings:
    _settings={}
    @staticmethod
    def dealString(string):
        if '.' in string or 'e' in string:
            return float(string)
        elif '\'' in string:
            return string.replace('\'','')
        else:
            return int(string)
    def addKey(self,key,value):
        self._settings[key]=value
    def getString(self,key):
        return self._settings[key].replace('\'','')
    def getNum(self,key):
        if '.' in self._settings[key] or 'e' in self._settings[key]:
            return float(self._settings[key])
        else:
            return int(self._settings[key])
    def getList(self,key):
        return  map(Settings.dealString,self._settings[key].replace('[','').replace(']','').split(','))
    def getDict(self,key):
        dic = {}
        tmp = self._settings[key].replace('{','').replace('}','').split(';')
        for iterm in tmp:
            dic[Settings.dealString(iterm.split(':')[0])]=Settings.dealString(iterm.split(':')[1])
        return dic
    def getValue(self,key):
        tmp = self._settings[key]
        if '{' in tmp and '}' in tmp:
            return self.getDict(key)
        elif '[' in tmp and ']' in tmp:
            return self.getList(key)
        elif "'" in tmp :
            tmp = tmp.replace("'",'')
            if tmp == 'True':
                return True
            elif tmp == 'False':
                return False
            elif tmp == 'None':
                return None
            else:
                return self.getString(key)
        else:
            return self.getNum(key)
    def __init__(self,path):
        pattern=r"^//"
        with open(path,'r') as f:
            settings = f.read()
            settings = settings.split('\n')
            for line in settings:
                if re.search(pattern,line) or (len(line) == 0):
                    continue
                line=line.split(' ')
                self._settings[line[0]]=line[2]