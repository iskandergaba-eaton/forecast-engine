import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from compy import Component, Output

class DataSource(Component):
    """
    A simple data source.

    This sends injected data to its output.

    Outputs:
        0: data passed to the inject function.
    """

    def __init__(self, config):
        """
        Initialise the data source.

        Args:
            config: configuration for the data sink
        """
        self.config = config
        super(DataSource, self).__init__()
        self.outputs.append(Output("DataSource(" + config + ").output"))
        print("New DataSource subscribed to:", config)

    def inject(self, data):
        """
        Simulate data received from the data source.

        Call this to make it appear that data has been received from
        somewhere. The data passed will by sent to the output.

        Args:
            data: the data to inject
        """
        print("DataSource(" + self.config + ") received:", data)
        self.outputs[0].send(data)
