def run_example():
    """
    Python ComPy Demo
    
    Demonstrates a network of 3 components as follows.

        DataSource --> Algorithm --> DataSink
    """

    from datasource import DataSource
    from datasink import DataSink
    from algorithm import Algorithm

    print("*******************************************")
    print("* Python ComPy Demo")
    print("*******************************************")

    print("")
    print("*** Create components")
    weather_source = DataSource("Weather")
    predictor = Algorithm()
    prediction_sink = DataSink("Predicted-Power")

    print("")
    print("*** Connect components")
    weather_source.outputs[0].connect_to(predictor.inputs[0])
    predictor.outputs[0].connect_to(prediction_sink.inputs[0])

    print("")
    print("*** Push some data into the weather source and see what happens")
    weather_source.inject("Some weather")

if __name__ == '__main__':
    run_example()