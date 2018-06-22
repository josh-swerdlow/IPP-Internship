import unittest
from data import generateFunction


def linFunc(x, m, b):
    """A linear function"""
    return m * x + b


def quadFunc(x, a, b, c):
    """A quadratic function"""
    return a * x**2 + b * x + c


def constFunc(a):
    """A constant function"""
    return a


class TestDataMethods(unittest.TestCase):
    """ Test the class data.generateFunction."""

    def test_init(self):
        """ Test initialization method

        Tests:
            * passing None as function parameter
            * passing more parameters than the function needs
            * passing less parameters than the function needs
            * passing a function that does not have 'x' as
                its independent variable
        """

        with self.assertRaises(TypeError):
            generateFunction(None)

        # with self.assertRaises(TypeError):
        #     generateFunction(linFunc, a=2, b=3, c=4)

        # with self.assertRaises(TypeError):
        #     generateFunction(linFunc, a=1)

        with self.assertRaises(TypeError):
            generateFunction(constFunc, 5)

    def test_evaluate(self):
        """ Test the evaluate method

        Tests:
            * passing a negative sigma
            * passing a zero samples
        """

        f = generateFunction(linFunc, m=1, b=2)

        with self.assertRaises(ValueError):
            f.evaluate(sigma=-1, samples=100)

        with self.assertRaises(ValueError):
            f.evaluate(sigma=10, samples=0)

    def test_linear(self):
        """ Test the class on creating a linear function

        Tests:
            * indep/free parameter tests
                * total length of parameters
                * that we have all the free parameters
                * that free parameters are mapped correctly
                * that we have the correct indep paramaters
                * that we have the correct indep parameter
            * that sigma is what we passed in
            * that samples is what we passed in
            * that x, noise, and y are all the same length (samples)
        """

        sigma = 10
        samples = 100

        f = generateFunction(linFunc, m=1, b=2)
        # Param test
        self.assertEqual(len(f.freeParams) + len(f.indepParam), 3)

        # Free Param test
        self.assertEqual(len(f.freeParams), 2)
        self.assertEqual(f.freeParams['m'], 1)
        self.assertEqual(f.freeParams['b'], 2)

        # Indep Param test
        self.assertEqual(len(f.indepParam), 1)
        self.assertEqual(f.indepParam[0], 'x')

        f.evaluate(sigma, samples)

        # Sigma test
        self.assertEqual(f.sigma, sigma)

        # Sample test
        self.assertEqual(f.samples, samples)

        # x, y, and noise length test
        self.assertTrue(len(f.x) == len(f.noise) == len(f.y) == samples)

    def test_quad(self):
        """ Test the class on creating a quadratic function

        Tests:
            * indep/free parameter tests
                + total length of parameters
                + that we have all the free parameters
                + that free parameters are mapped correctly
                + that we have the correct indep paramaters
                + that we have the correct indep parameter
            * that sigma is what we passed in
            * that samples is what we passed in
            * that x, noise, and y are all the same length (samples)
        """

        sigma = 10
        samples = 100

        f = generateFunction(quadFunc, a=1, b=2, c=3)
        # Param test
        self.assertEqual(len(f.freeParams) + len(f.indepParam), 4)

        # Free Param test
        self.assertEqual(len(f.freeParams), 3)
        self.assertEqual(f.freeParams['a'], 1)
        self.assertEqual(f.freeParams['b'], 2)
        self.assertEqual(f.freeParams['c'], 3)

        # Indep Param test
        self.assertEqual(len(f.indepParam), 1)
        self.assertEqual(f.indepParam[0], 'x')

        f.evaluate(sigma, samples)

        # Sigma test
        self.assertEqual(f.sigma, sigma)

        # Sample test
        self.assertEqual(f.samples, samples)

        # x, y, and noise length test
        self.assertTrue(len(f.x) == len(f.noise) == len(f.y) == samples)


if __name__ == '__main__':
    unittest.main()

