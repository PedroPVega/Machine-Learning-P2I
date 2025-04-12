using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

class NeuralNetwork
{
    public int InputSize = 784;
    public int OutputSize = 10;
    public int HiddenSize;
    public int NumHiddenLayers;
    public double LearningRate = 0.01;

    List<double[,]> Weights = new List<double[,]>();
    List<double[]> Biases = new List<double[]>();
    Random rand = new Random();

    public NeuralNetwork(int hiddenSize, int numHiddenLayers)
    {
        this.HiddenSize = hiddenSize;
        this.NumHiddenLayers = numHiddenLayers;

        // Input to first hidden layer
        Weights.Add(InitWeights(InputSize, hiddenSize));
        Biases.Add(new double[hiddenSize]);

        // Hidden layers
        for (int i = 1; i < numHiddenLayers; i++)
        {
            Weights.Add(InitWeights(hiddenSize, hiddenSize));
            Biases.Add(new double[hiddenSize]);
        }

        // Last hidden layer to output
        Weights.Add(InitWeights(hiddenSize, OutputSize));
        Biases.Add(new double[OutputSize]);
    }

    double[,] InitWeights(int inSize, int outSize)
    {
        double[,] layer = new double[inSize, outSize];
        for (int i = 0; i < inSize; i++)
            for (int j = 0; j < outSize; j++)
                layer[i, j] = rand.NextDouble() * 0.01;
        return layer;
    }

    double[] ReLU(double[] x) => x.Select(v => Math.Max(0, v)).ToArray();
    double[] ReLUDerivative(double[] x) => x.Select(v => v > 0 ? 1.0 : 0.0).ToArray();

    double[] Softmax(double[] x)
    {
        double max = x.Max();
        double[] exps = x.Select(v => Math.Exp(v - max)).ToArray();
        double sum = exps.Sum();
        return exps.Select(v => v / sum).ToArray();
    }

    // Forward pass
    public (List<double[]> activations, List<double[]> zs) Forward(double[] input)
    {
        List<double[]> activations = new List<double[]> { input };
        List<double[]> zs = new List<double[]>();

        double[] current = input;

        for (int l = 0; l < NumHiddenLayers; l++)
        {
            double[] z = MatVec(Weights[l], current).Zip(Biases[l], (a, b) => a + b).ToArray();
            zs.Add(z);
            current = ReLU(z);
            activations.Add(current);
        }

        // Output layer
        int last = Weights.Count - 1;
        double[] zOut = MatVec(Weights[last], current).Zip(Biases[last], (a, b) => a + b).ToArray();
        zs.Add(zOut);
        double[] output = Softmax(zOut);
        activations.Add(output);

        return (activations, zs);
    }

    // Backward pass
    public void Backward(double[] input, int label)
    {
        var (activations, zs) = Forward(input);
        double[] target = new double[OutputSize];
        target[label] = 1.0;

        double[] delta = activations.Last().Zip(target, (o, t) => o - t).ToArray(); // Cross-entropy + softmax gradient

        for (int l = Weights.Count - 1; l >= 0; l--)
        {
            double[] aPrev = activations[l];
            double[,] wGrad = OuterProduct(aPrev, delta);
            double[] bGrad = delta;

            // Gradient descent
            for (int i = 0; i < Weights[l].GetLength(0); i++)
                for (int j = 0; j < Weights[l].GetLength(1); j++)
                    Weights[l][i, j] -= LearningRate * wGrad[i, j];

            for (int j = 0; j < Biases[l].Length; j++)
                Biases[l][j] -= LearningRate * bGrad[j];

            if (l > 0)
            {
                double[] z = zs[l - 1];
                double[] dReLU = ReLUDerivative(z);
                delta = DotTrans(Weights[l], delta).Zip(dReLU, (a, b) => a * b).ToArray();
            }
        }
    }

    // Helper math
    double[] MatVec(double[,] mat, double[] vec)
    {
        int rows = mat.GetLength(0), cols = mat.GetLength(1);
        double[] result = new double[cols];
        for (int j = 0; j < cols; j++)
            for (int i = 0; i < rows; i++)
                result[j] += mat[i, j] * vec[i];
        return result;
    }

    double[,] OuterProduct(double[] a, double[] b)
    {
        double[,] result = new double[a.Length, b.Length];
        for (int i = 0; i < a.Length; i++)
            for (int j = 0; j < b.Length; j++)
                result[i, j] = a[i] * b[j];
        return result;
    }

    double[] DotTrans(double[,] mat, double[] vec)
    {
        int rows = mat.GetLength(0), cols = mat.GetLength(1);
        double[] result = new double[rows];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i] += mat[i, j] * vec[j];
        return result;
    }

    // Load MNIST CSV
    public static List<(double[], int)> LoadMNIST(string filePath, int maxSamples = 6000)
    {
        var data = new List<(double[], int)>();
        var lines = File.ReadLines(filePath).Skip(1).Take(maxSamples);
        foreach (var line in lines)
        {
            var parts = line.Split(',').Select(int.Parse).ToArray();
            int label = parts[0];
            double[] pixels = parts.Skip(1).Select(v => v / 255.0).ToArray();
            data.Add((pixels, label));
        }
        return data;
    }
}


