import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from compy import Component, Input

class DataSink(Component):
    """
    A simple data sink.

    This takes data and outputs it on the console.

    Inputs:
        0: any data
    """

    def __init__(self, config):
        """
        Initialise the data sink.

        Args:
            config: configuration for the data sink
        """
        self.config = config
        super(DataSink, self).__init__()
        self.inputs.append(Input("DataSink(" + config + ").input",
                                 self._publish))
        print("New DataSink publishing to:", config)

    def _publish(self, data):
        """
        Handle data from the sink input.

        Args:
            data: the data from the sink input
        """
        print("Datasink(" + self.config + ") sending:", data)
