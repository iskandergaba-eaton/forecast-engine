"""
ComPy Component Framework for Python
------------------------------------

The is the Eaton CIP Python Component Framework for packaging
algorithms and other code in a standard way. It makes interfacing
easier and greatly simplifies build applications and system from
Python modules.

Copyright (C) 2021 Eaton Corporation Inc.
All rights reserved.

Unauthorized copying of this material, via any medium is strictly
prohibited without the express permission of Eaton Corporation Inc.
"""

class Component:        # pylint: disable=too-few-public-methods
    """
    Base component class for Intelligence Component Framework for Python.
    This is the base case for every component and enables the framework
    to apply a consistent interface for any component.

    To create a component, use this as the base class and in the subclass
    constructor call the Component constructor then create Inputs and
    Outputs and append them to inputs and outputs. If the component
    requires an ongoing process, is should override the process()
    function.

    Example component:

        class ADataSource(Component):

            def __init__(self, config):
                super(ADataSource, self).__init__()
                self.config = config
                self.outputs.append(Output("ADataSource.output"))

            def process(self):
                # Could wait for data here and loop to repeat

    """

    def __init__(self):
        """
        Initialise the component base internals.

        The component subclass calls this before its own
        initialisation.
        """
        self.inputs = []
        self.outputs = []
        

    def process(self):
        """
        The component subclass overrides this if it requires an
        ongoing process.
        """

class Input:            # pylint: disable=too-few-public-methods
    """
    The class used for inputs of a class Component.
    """

    def __init__(self, name, receive_func):
        """
        Initialise the input.

        Args:
            name: the name of the input
            receive_func: the function that will process data received
                on the input
        """

        self.name = name
        self.receive = receive_func


class Output:
    """
    The class used for outputs of a class Component.
    """

    def __init__(self, name):
        """
        Initialise the output.

        Args:
            name: the name of the output
        """
        self.name = name
        self.inputs = []

    def connect_to(self, _input):
        """
        Connect this output to an input.

        Adds the input to the list of connections for this output. The
        input must be a class Input or a subclass of class Input.

        Args:
            _input: the Input to be connected.
        """
        self.inputs.append(_input)
        print("Connecting", self.name, "to", _input.name)

    def send(self, data):
        """
        Send data out.

        A component calls this to send data out of this output. It
        sends the data to all the connected inputs.

        Args:
            data: the data to be sent.
        """
        for _input in self.inputs:
            _input.receive(data)
