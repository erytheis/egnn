import wntr



class EpanetDummySimulator( wntr.sim.EpanetSimulator):
    """
    This is a child class to WNTRSimulator that is used for loading and initializing the .INP file and extracting options
    :param epanet_filename: str
    """


    def __init__(self, epanet_filename):
        self.input_fullpath = epanet_filename
        self.wn = wntr.network.WaterNetworkModel(self.input_fullpath)
        wntr.sim.EpanetSimulator.__init__(self,self.wn)



