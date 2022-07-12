import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from compy import Component, Input, Output

class Algorithm(Component):
    """
    Dummy Power generation prediction algorithm component.

    This takes in weather data and outputs text "A prediction".

    Inputs:
        0: weather data
    Outputs:
        0: predicted generated power
    """

    def __init__(self):
        """
        Initialise the power predictor.
        """
        super(Algorithm, self).__init__()
        self.inputs.append(Input("Algorithm.weather_in", 
                                 self._handle_weather))
        self.outputs.append(Output("Algorithm.prediction_out"))
        print("New Algorithm")

    def _handle_weather(self, weather):
        """
        Handle weather from the weather input.

        Each time weather is passed in it will result in a prediction
        being sent to the output.
        Args:
            weather: the weather to be processed 
        """
        print("Algorithm handling weather:", weather)
        prediction = "A prediction"
        print("Algorithm sending:", prediction)
        self.outputs[0].send(prediction)
