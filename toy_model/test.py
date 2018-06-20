from model import initData


def test():

    # Initialize function to be fit #
    # def linFunc(x, m, b):
    #     return m * x + b
    def quadFunc(x, a, b, c):
        return a * x**2 + b * x + c

    # Initialize some data into a DataFrame #
    # Parameters for data
    SAMPLES = 100
    sigma = 1

    # Parameters for function
    # m = 1
    # b = 2
    a = 1
    b = 0.5
    c = 5
    params = [a, b, c]

    data = initData(lambda x: quadFunc(x, *params), SAMPLES, sigma)

    data.plot()


if __name__ == 'main':
    test()
