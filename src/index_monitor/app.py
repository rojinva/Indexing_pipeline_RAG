from factory.monitor import Index



class IndexMonitor:
    scheduler = None
    index = None
    
    def __init__(self):
        self.index = Index()
        
    def letsMonitor(self):
        return self.index.indexerChecks()
    
    
    

if __name__ == '__main__':
    letsSee = IndexMonitor()
    letsSee.letsMonitor()
